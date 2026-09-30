// 02_main_arm.cpp
// Lesson 20 实验 2: 在板上跑 4 个 TVM matmul 版本, 测耗时和加速比
//
// 不用 graph_executor(实验 02 走 tvm.build(PrimFunc) 路径,
// .so 没有 __tvm_dev_mblob, graph_executor 加载会 segfault)
// 改用 packed_func 直接调用: .so 里的 "default" 签名是 (A, B, C) -> void
//
// 编译: make
// 板子跑: ./02_main_arm libmatmul_naive.so [n_iter=10]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>

#define CHECK(expr)                                             \
    do                                                          \
    {                                                           \
        int _r = (expr);                                        \
        if (_r)                                                 \
        {                                                       \
            fprintf(stderr, "TVM error: %s = %d\n", #expr, _r); \
            exit(1);                                            \
        }                                                       \
    } while (0)

#define M 256
#define N 256
#define K 256

static double now_sec(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

static void random_fill(float *p, int n, unsigned seed)
{
    srand(seed);
    for (int i = 0; i < n; ++i)
        p[i] = (float)rand() / RAND_MAX - 0.5f;
}

// 参考实现：三层朴素循环matmul（C标量）
static void matmul_ref(const float *A,
                       const float *B,
                       float *C)
{
    memset(C, 0, sizeof(float) * M * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <libmatmul_X.so> [n_iter=10]\n", argv[0]);
        return 1;
    }
    const char *so_path = argv[1];
    int n_iter = (argc >= 3) ? atoi(argv[2]) : 10;
    double t0, t1;
    // 1. 准备数据
    static float A_buf[M * K];
    static float B_buf[K * N];
    static float C_buf[M * N];
    static float C_ref[M * N];
    random_fill(A_buf, M * K, 1);
    random_fill(B_buf, K * N, 2);
    printf("[*] M=N=K=256, 算 %d 次取平均\n", n_iter);

    // 2. 加载两个 so：基准 naive + 被测模块
    // ⚠️ 关键：加速比必须在"同一进程内"对比——板子无法锁频，
    //    每个进程启动时的 CPU 频率状态不同，跨进程比 ref 会虚高 3~4 倍
    const char *base_path = "libmatmul_naive.so";  // 基准：naive 在同一目录

    TVMModuleHandle base_mod = NULL;
    CHECK(TVMModLoadFromFile(base_path, "so", &base_mod));
    TVMFunctionHandle base_func = NULL;
    CHECK(TVMModGetFunction(base_mod, "default", 0, &base_func));

    TVMModuleHandle mod = NULL;
    CHECK(TVMModLoadFromFile(so_path, "so", &mod));
    printf("[*] loaded %s\n", so_path);

    TVMFunctionHandle func = NULL;
    CHECK(TVMModGetFunction(mod, "default", 0, &func));
    printf("[*] got packed func %p\n", func);

    // 3. 把A/B/C封装成DLTensor
    int64_t a_shape[2] = {M, K};
    int64_t b_shape[2] = {K, N};
    int64_t c_shape[2] = {M, N};
    DLTensor A;
    A.data = A_buf;
    A.device = (DLDevice){kDLCPU, 0};
    A.ndim = 2;
    A.dtype = (DLDataType){kDLFloat, 32, 1};
    A.shape = a_shape;
    A.strides = NULL;
    A.byte_offset = 0;
    DLTensor B;
    B.data = B_buf;
    B.device = (DLDevice){kDLCPU, 0};
    B.ndim = 2;
    B.dtype = (DLDataType){kDLFloat, 32, 1};
    B.shape = b_shape;
    B.strides = NULL;
    B.byte_offset = 0;
    DLTensor C;
    C.data = C_buf;
    C.device = (DLDevice){kDLCPU, 0};
    C.ndim = 2;
    C.dtype = (DLDataType){kDLFloat, 32, 1};
    C.shape = c_shape;
    C.strides = NULL;
    C.byte_offset = 0;
    // 4. 参考实现测时已挪到 warmup 之后（见下方步骤 7）
    //    原因：板子无法锁频，CPU 频率要跑几次才拉满，
    //         频率没起来时测 ref 会虚高 3~4 倍，加速比失真

    // 5. 通用调用 lambda：给任意 TVMFunctionHandle 执行一次 (A,B,C) -> void
    //    [&] = 按引用捕获外面的 A/B/C 变量，lambda 内部直接可用
    //    （之前大段讲解已并入 SUMMARY.md，这里只留核心）
    auto call_with = [&](TVMFunctionHandle f)
    {
        TVMValue args[3];
        int tcodes[3];
        args[0].v_handle = &A;
        tcodes[0] = kTVMDLTensorHandle;
        args[1].v_handle = &B;
        tcodes[1] = kTVMDLTensorHandle;
        args[2].v_handle = &C;
        tcodes[2] = kTVMDLTensorHandle;
        TVMValue r;
        int rt;
        CHECK(TVMFuncCall(f, args, tcodes, 3, &r, &rt));
    };
    // call_tvm = 固定调用被测模块（正确性检查 + warmup 用）
    auto call_tvm = [&]() { call_with(func); };

    // 6. CPU 频率预热：先跑几次，把频率拉到最高
    //    （板子无法锁频，程序刚启动时 CPU 在低频，直接测会虚高 3~4 倍）
    for (int i = 0; i < 3; ++i)
        call_tvm();
    for (int i = 0; i < 3; ++i)
        call_with(base_func);

    // 7. 参考实现测时 —— 3 轮取 min（频率已拉满，仅供参考）
    double best_ref_ms = 1e9;
    for (int round = 0; round < 3; ++round)
    {
        t0 = now_sec();
        matmul_ref(A_buf, B_buf, C_ref);
        t1 = now_sec();
        double ms = (t1 - t0) * 1000;
        if (ms < best_ref_ms) best_ref_ms = ms;
    }
    double t_ref = best_ref_ms / 1e3;
    printf("[*] 参考实现 = %.2f ms (3 轮取 min)\n", best_ref_ms);

    // 正确性检查：用被测模块算一次，对比 C_ref（C 标量参考）
    call_tvm();
    float max_diff = 0;
    for (int i = 0; i < M * N; ++i)
    {
        float d = C_buf[i] - C_ref[i];
        if (d < 0)
            d = -d;
        if (d > max_diff)
            max_diff = d;
    }
    printf("[*] max_diff=%g %s\n", max_diff, (max_diff < 1e-3) ? "OK" : "FAIL");

    // 8. 通用测时：跑 n_iter 次, 3 轮取最小, 返回 best ms
    auto bench = [&](TVMFunctionHandle f) -> double {
        double best = 1e9;
        for (int round = 0; round < 3; ++round)
        {
            t0 = now_sec();
            for (int i = 0; i < n_iter; ++i)
            {
                TVMValue args[3];
                int tcodes[3];
                args[0].v_handle = &A; tcodes[0] = kTVMDLTensorHandle;
                args[1].v_handle = &B; tcodes[1] = kTVMDLTensorHandle;
                args[2].v_handle = &C; tcodes[2] = kTVMDLTensorHandle;
                TVMValue r;
                int rt;
                CHECK(TVMFuncCall(f, args, tcodes, 3, &r, &rt));
            }
            t1 = now_sec();
            double ms = (t1 - t0) / n_iter * 1000;
            if (ms < best) best = ms;
        }
        return best;
    };

    // 9. 同进程对比：基准 naive vs 被测模块（频率状态完全相同，公平）
    double base_ms = bench(base_func);   // naive 基准
    double avg_ms = bench(func);         // 被测模块
    double gflops = 2.0 * M * N * K / (avg_ms / 1e3) / 1e9;
    double speedup = base_ms / avg_ms;   // vs naive 的加速比（公平）

    const char *tag = strrchr(so_path, '/');
    tag = tag ? tag + 1 : so_path;
    printf("[=] %-32s best=%.2f ms  GFLOPS=%.1f  max_diff=%g  (naive=%.2f ms, 加速=%.1fx, ref=%.2f ms)\n",
           tag, avg_ms, gflops, max_diff, base_ms, speedup, t_ref * 1000);

    TVMModFree(&mod);
    TVMModFree(&base_mod);
    return 0;
}
