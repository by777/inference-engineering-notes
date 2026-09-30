// main.c
// Lesson 17: TVM 部署 —— 用 C API 调用 MatMul 内核
//
// 与 Lesson 16 的关系:
//   Lesson 16: mlp3.onnx 模型图（3 层 MLP）
//   本实验:    matmul 算子（单算子）
//   共同点:    都是 graph_executor 路径，C 调用流程完全一致
//
// 调用流程（与 Lesson 16 完全一样）:
//   1. TVMModLoadFromFile        → 加载 .so
//   2. TVMModGetFunction(mod, "default", 0, &factory)
//   3. TVMFuncCall(factory, {cpu}, 1) → 得到 graph_executor 模块 gmod
//   4. TVMModGetFunction(gmod, "set_input"/"run"/"get_output", 1, ...)
//   5. set_input("A", tensor) → set_input("B", tensor) → run() → get_output(0, C)
//
// 编译: make
// 运行: ./main

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>

// 辅助宏: 检查 TVM C API 返回值（0 = 成功，非 0 = 错误）
#define CHECK(expr)                                  \
    do                                               \
    {                                                \
        int _ret = (expr);                           \
        if (_ret != 0)                               \
        {                                            \
            fprintf(stderr, "Error: %s (code %d)\n", \
                    #expr, _ret);                    \
            exit(1);                                 \
        }                                            \
    } while (0)

#define M 256
#define N 256
#define K 256

int main()
{
    // ---------- 1. 加载 .so ----------
    TVMModuleHandle mod = NULL;
    CHECK(TVMModLoadFromFile("./libmatmul.so", "so", &mod));
    printf("加载 libmatmul.so 成功\n");

    // ---------- 2. 创建 graph_executor ----------
    // graph_executor 是 TVM 的"图执行器"（类似 ORT 的 InferenceSession）
    // 它负责管理输入输出张量的生命周期、调度算子执行顺序
    //
    // 三步创建:
    //   ① 从 .so 里找 "default" 工厂函数
    //   ② 调工厂，传 CPU 设备参数 → 得到 gmod
    //   ③ 从 gmod 里找 set_input / run / get_output 三个函数
    TVMFunctionHandle factory = NULL;
    CHECK(TVMModGetFunction(mod, "default", 0, &factory)); // 0: default 在 .so 自己
    printf("拿到工厂函数 default\n");

    // 调工厂: 传 DLDevice 参数（CPU 设备 0）
    TVMValue fargs[1];
    int ftcodes[1];
    fargs[0].v_device = (DLDevice){kDLCPU, 0};
    ftcodes[0] = kDLDevice;
    TVMValue fret;
    int fret_tcode;
    CHECK(TVMFuncCall(factory, fargs, ftcodes, 1, &fret, &fret_tcode));
    TVMModuleHandle gmod = fret.v_handle; // gmod = graph_executor 模块
    printf("创建 graph_executor 成功\n");

    // 拿 set_input / run / get_output
    // query_imports=1: 这些函数在 graph_executor 内部模块里，不在 .so 本身
    TVMFunctionHandle set_input, run_func, get_output;
    CHECK(TVMModGetFunction(gmod, "set_input", 1, &set_input));
    CHECK(TVMModGetFunction(gmod, "run", 1, &run_func));
    CHECK(TVMModGetFunction(gmod, "get_output", 1, &get_output));

    // ---------- 3. 准备输入张量 A[M,K] B[K,N] ----------
    // 用简单模式填数据，方便手算验证
    float A_buf[M * K];
    float B_buf[K * N];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            A_buf[i * K + j] = (float)(i + j); // A[i][j] = i + j
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B_buf[i * N + j] = (float)(i - j); // B[i][j] = i - j

    // DLTensor: TVM 和 C 之间交换张量的统一格式（DLPack 标准）
    int64_t a_shape[2] = {M, K};
    int64_t b_shape[2] = {K, N};
    int64_t c_shape[2] = {M, N};

    DLTensor A = {
        .data = A_buf,              // 数据指针
        .device = {kDLCPU, 0},      // CPU 设备 0
        .ndim = 2,                  // 2 维
        .dtype = {kDLFloat, 32, 1}, // float32 标量
        .shape = a_shape,           // [M, K]
        .strides = NULL,            // NULL = 行主序连续内存
        .byte_offset = 0,           // 从第 0 字节开始
    };
    DLTensor B = {
        .data = B_buf,
        .device = {kDLCPU, 0},
        .ndim = 2,
        .dtype = {kDLFloat, 32, 1},
        .shape = b_shape,
        .strides = NULL,
        .byte_offset = 0,
    };
    float C_buf[M * N]; // 输出 buffer
    DLTensor C = {
        .data = C_buf,
        .device = {kDLCPU, 0},
        .ndim = 2,
        .dtype = {kDLFloat, 32, 1},
        .shape = c_shape,
        .strides = NULL,
        .byte_offset = 0,
    };

    // ---------- 4. 推理: set_input → run → get_output ----------
    // set_input("A", tensor): 把 A 张量注册到 graph_executor 的输入槽
    {
        TVMValue args[2];
        int tcodes[2];
        args[0].v_str = "A"; // 输入名（对应 Relay 里的 relay.var("A", ...)）
        tcodes[0] = kTVMStr;
        args[1].v_handle = &A; // 张量指针
        tcodes[1] = kTVMDLTensorHandle;
        TVMValue ret;
        int ret_tcode;
        CHECK(TVMFuncCall(set_input, args, tcodes, 2, &ret, &ret_tcode));
    }
    // set_input("B", tensor)
    {
        TVMValue args[2];
        int tcodes[2];
        args[0].v_str = "B";
        tcodes[0] = kTVMStr;
        args[1].v_handle = &B;
        tcodes[1] = kTVMDLTensorHandle;
        TVMValue ret;
        int ret_tcode;
        CHECK(TVMFuncCall(set_input, args, tcodes, 2, &ret, &ret_tcode));
    }
    printf("设置输入完成\n");

    // run: 执行整个计算图（这里就是 nn.matmul）
    {
        TVMValue ret;
        int ret_tcode;
        CHECK(TVMFuncCall(run_func, NULL, NULL, 0, &ret, &ret_tcode));
    }
    printf("推理完成\n");

    // get_output(0, C): 取第 0 个输出到 C 张量
    {
        TVMValue args[2];
        int tcodes[2];
        args[0].v_int64 = 0; // 输出索引 0
        tcodes[0] = kTVMArgInt;
        args[1].v_handle = &C; // 输出张量指针
        tcodes[1] = kTVMDLTensorHandle;
        TVMValue ret;
        int ret_tcode;
        CHECK(TVMFuncCall(get_output, args, tcodes, 2, &ret, &ret_tcode));
    }

    // ---------- 5. 验证 ----------
    // 手算几个点: C[i][j] = Σ_k (i+k) * (k-j)
    // 用 float64 手算（避免 float32 累积误差），和内核输出对比
    int errors = 0;
    for (int i = 0; i < M; i += 32) // 抽样验证（256² 全查太啰嗦）
    {
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
                    printf("  Mismatch C[%d][%d] = %.4f, 期望 %.4f\n", i, j, got, (float)ref);
                errors++;
            }
        }
    }
    if (errors == 0)
        printf("✅ 验证通过: 抽样点全部与手算一致\n");
    else
        printf("❌ %d 个点不一致\n", errors);

    printf("输出 C[0][0..7]: ");
    for (int j = 0; j < 8; j++)
        printf("%.2f ", C_buf[j]);
    printf("\n");

    // ---------- 6. 清理 ----------
    CHECK(TVMFuncFree(get_output));
    CHECK(TVMFuncFree(run_func));
    CHECK(TVMFuncFree(set_input));
    CHECK(TVMModFree(gmod));
    CHECK(TVMFuncFree(factory));
    CHECK(TVMModFree(mod));
    printf("完成\n");
    return 0;
}