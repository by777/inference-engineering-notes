// main_arm.c
// Lesson 18: 板子上运行的 C 程序 —— 调用 TVM 编译的 arm64 MatMul
//
// 和 Lesson 17 实验 3 的 main.c 完全一样，区别只是:
//   加载的 .so 是 libmatmul_arm.so（arm64 版）
//   链接的 runtime 是 arm64 版（tvm-bin-arm/）
//
// 调用流程: TVMModLoadFromFile → default 工厂 → set_input/run/get_output
#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(expr)                                         \
    do                                                      \
    {                                                       \
        int _r = (expr);                                    \
        if (_r)                                             \
        {                                                   \
            fprintf(stderr, "Error: %s (%d)\n", #expr, _r); \
            exit(1);                                        \
        }                                                   \
    } while (0)
#define M 256
#define N 256
#define K 256

int main()
{
    // 加载 arm64 版 .so
    TVMModuleHandle mod = NULL;
    CHECK(TVMModLoadFromFile("./libmatmul_arm.so", "so", &mod));
    printf("加载 libmatmul_arm.so 成功\n");

    // 创建 graph_executor（default 工厂 → gmod）
    TVMFunctionHandle factory = NULL;
    CHECK(TVMModGetFunction(mod, "default", 0, &factory));
    TVMValue fargs[1];
    int ftcodes[1];
    fargs[0].v_device = (DLDevice){kDLCPU, 0};
    ftcodes[0] = kDLDevice;
    TVMValue fret;
    int fret_tcode;
    CHECK(TVMFuncCall(factory, fargs, ftcodes, 1, &fret, &fret_tcode));
    TVMModuleHandle gmod = fret.v_handle;
    printf("创建 graph_executor 成功\n");

    // 拿 set_input / run / get_output
    TVMFunctionHandle set_input, run_func, get_output;
    CHECK(TVMModGetFunction(gmod, "set_input", 1, &set_input));
    CHECK(TVMModGetFunction(gmod, "run", 1, &run_func));
    CHECK(TVMModGetFunction(gmod, "get_output", 1, &get_output));

    // 准备数据: A[i][j]=i+j, B[i][j]=i-j
    float A_buf[M * K], B_buf[K * N], C_buf[M * N];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            A_buf[i * K + j] = (float)(i + j);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B_buf[i * N + j] = (float)(i - j);
    int64_t a_shape[2] = {M, K}, b_shape[2] = {K, N}, c_shape[2] = {M, N};
    DLTensor A = {.data = A_buf, .device = {kDLCPU, 0}, .ndim = 2, .dtype = {kDLFloat, 32, 1}, .shape = a_shape, .strides = NULL, .byte_offset = 0};
    DLTensor B = {.data = B_buf, .device = {kDLCPU, 0}, .ndim = 2, .dtype = {kDLFloat, 32, 1}, .shape = b_shape, .strides = NULL, .byte_offset = 0};
    DLTensor C = {.data = C_buf, .device = {kDLCPU, 0}, .ndim = 2, .dtype = {kDLFloat, 32, 1}, .shape = c_shape, .strides = NULL, .byte_offset = 0};

    // set_input("A"/"B")
    {
        TVMValue args[2];
        int tc[2];
        args[0].v_str = "A";
        tc[0] = kTVMStr;
        args[1].v_handle = &A;
        tc[1] = kTVMDLTensorHandle;
        TVMValue r;
        int rt;
        CHECK(TVMFuncCall(set_input, args, tc, 2, &r, &rt));
    }
    {
        TVMValue args[2];
        int tc[2];
        args[0].v_str = "B";
        tc[0] = kTVMStr;
        args[1].v_handle = &B;
        tc[1] = kTVMDLTensorHandle;
        TVMValue r;
        int rt;
        CHECK(TVMFuncCall(set_input, args, tc, 2, &r, &rt));
    }
    printf("设置输入完成\n");

    // run
    {
        TVMValue r;
        int rt;
        CHECK(TVMFuncCall(run_func, NULL, NULL, 0, &r, &rt));
    }
    printf("推理完成\n");

    // get_output
    {
        TVMValue args[2];
        int tc[2];
        args[0].v_int64 = 0;
        tc[0] = kTVMArgInt;
        args[1].v_handle = &C;
        tc[1] = kTVMDLTensorHandle;
        TVMValue r;
        int rt;
        CHECK(TVMFuncCall(get_output, args, tc, 2, &r, &rt));
    }

    // 验证（抽样点手算对比）
    int errors = 0;
    for (int i = 0; i < M; i += 32)
        for (int j = 0; j < N; j += 32)
        {
            double ref = 0.0;
            for (int k = 0; k < K; k++)
                ref += (double)(i + k) * (double)(k - j);
            float got = C_buf[i * N + j];
            double err = ref != 0 ? (got - ref) / ref : (got - ref);
            if (err > 1e-4 || err < -1e-4)
            {
                if (errors < 5)
                    printf("  Mismatch C[%d][%d]=%.4f 期望%.4f\n", i, j, got, (float)ref);
                errors++;
            }
        }
    if (errors == 0)
        printf("✅ 验证通过: 抽样点全部与手算一致 (在 ARM 板子上!)\n");
    else
        printf("❌ %d 个点不一致\n", errors);

    printf("C[0][0..7]: ");
    for (int j = 0; j < 8; j++)
        printf("%.2f ", C_buf[j]);
    printf("\n完成\n");
    return 0;
}