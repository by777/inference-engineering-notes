// main.c
// Lesson 16: TVM 部署进阶 —— 纯 C API 调用（对应 Lesson 08 的 ORT C API）
//
// 对比 Lesson 15 (C++): 去掉 C++ 的 Module/PackedFunc 包装，用纯 C 函数
// 对比 Lesson 08 (ORT C API): 同样是用 C 函数加载模型 + 推理，不同引擎
//
// C API 核心函数:
//   TVMModLoadFromFile  → 加载 .so（类似 CreateSession）
//   TVMModGetFunction   → 按名字拿函数（类似 SessionGetFunc + 动态调用）
//   TVMFuncCall         → 调用函数（类似 Run，但具体这一步自底向上构成调用）
//   DLTensor            → 张量交换格式（类似 OrtValue）
//
// 编译: gcc -std=c17 -I../tvm-src/include -I../tvm-src/3rdparty/dlpack/include
//            -I../tvm-src/3rdparty/dmlc-core/include
//            -L../tvm-bin -ltvm_runtime -ldl -lpthread
//            -Wl,-rpath,../tvm-bin -o main main.c
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

int main()
{
    // TVM 的 C API 只有 3 个核心函数：
    // TVMModLoadFromFile   → 加载 .so
    // TVMModGetFunction    → 从 .so 里拿一个"名字对应的函数"
    // TVMFuncCall          → 调用那个函数（传入参数 + 类型码）

    // ---------- 1. 加载模型 ----------
    // 对比 ORT: CreateSession(env, "model.onnx", ...)
    TVMModuleHandle mod = NULL;
    CHECK(TVMModLoadFromFile("./libmlp3.so", "so", &mod)); // 核心函数
    printf("加载 libmlp3.so 成功\n");

    // ---------- 2. 创建 graph executor ----------
    // ORT 的做法（一步到位）：
    //   session 加载完后自带执行能力，直接调 Run

    // TVM 的做法（三步）：
    //   ① 从 .so 里找一个叫 "default" 的函数 → factory
    //   ② 调用 factory，传入 CPU 设备参数 → 得到一个"图模块" gmod
    //   ③ 从 gmod 里找 set_input / run / get_output 三个函数

    // 2a. 拿 "default" 工厂函数 → 类似 lib["default"]
    TVMFunctionHandle factory = NULL;
    // 0 = 只查自己（快，但查不到就返回 NULL）
    // 1 = 自己找不到时，连带查导入链（慢一点，但更全）
    // 函数定义在"自己身上"传 0，定义在"导入的子模块"传 1；拿不准就传 1。
    CHECK(TVMModGetFunction(mod, "default", 0, &factory)); // 核心函数

    // 2b. 创建 CPU 设备参数
    TVMValue factory_args[1];
    int factory_tcodes[1];
    factory_args[0].v_device = (DLDevice){kDLCPU, 0};
    factory_tcodes[0] = kDLDevice;

    TVMValue factory_ret;
    int factory_ret_tcode;
    CHECK(TVMFuncCall(factory, factory_args, factory_tcodes, 1,
                      &factory_ret, &factory_ret_tcode)); // 核心函数
    TVMModuleHandle gmod = factory_ret.v_handle;

    // 2c. 拿 set_input / run / get_output 三个函数（query_imports=1）
    TVMFunctionHandle set_input, run_func, get_output;
    CHECK(TVMModGetFunction(gmod, "set_input", 1, &set_input));
    CHECK(TVMModGetFunction(gmod, "run", 1, &run_func));
    CHECK(TVMModGetFunction(gmod, "get_output", 1, &get_output));

    // ---------- 3. 准备输入 X[1,256] float32 全 0.5 ----------
    float input_buf[1 * 256];
    for (int i = 0; i < 256; i++)
        input_buf[i] = 0.5f;

    int64_t input_shape[2] = {1, 256};
    DLTensor input = {
        .data = input_buf,
        .device = {kDLCPU, 0}, // CPU第 0 个 CPU
        .ndim = 2,
        .dtype = {kDLFloat, 32, 1}, //   └─code数据类型大类─┘ └bits每个元素的位宽┘ └lanes向量通道数1 = 标量（普通 float）┘
        .shape = input_shape,
        .strides = NULL,  // "按标准行优先规则自动算步长"
        .byte_offset = 0, // 从 buffer 的第几个字节开始读
    };

    // ---------- 4. 推理 ----------
    // set_input —— 类似 m.set_input("X", tensor)
    {
        TVMValue args[2];
        int tcodes[2];
        args[0].v_str = "X";
        tcodes[0] = kTVMStr; // "这是字符串"
        args[1].v_handle = &input;
        tcodes[1] = kTVMDLTensorHandle; // "这是张量指针"
        TVMValue ret_val;
        int ret_tcode;
        CHECK(TVMFuncCall(set_input, args, tcodes, 2, &ret_val, &ret_tcode));
        //         // ORT:
        // Run(session, NULL, input_names, inputs, 1, output_names, 1, &outputs);

        // // TVM:
        // TVMFuncCall(set_input, args, tcodes, 2, NULL, NULL);
        // //           │          │     │       │    │
        // //           │          │     │       参数个数 不需要返回值
        // //           │          参数数组  类型码数组
        // //           要调的函数 "set_input"
    }

    // run —— 类似 m.run()
    {
        TVMValue ret_val;
        int ret_tcode;
        CHECK(TVMFuncCall(run_func, NULL, NULL, 0, &ret_val, &ret_tcode));
    }

    // get_output —— 类似 m.get_output(0, out)
    {
        float output_buf[1 * 10] = {0};
        int64_t output_shape[2] = {1, 10};
        DLTensor output = {
            .data = output_buf,
            .device = {kDLCPU, 0},
            .ndim = 2,
            .dtype = {kDLFloat, 32, 1},
            .shape = output_shape,
            .strides = NULL,
            .byte_offset = 0,
        };

        TVMValue args[2];
        int tcodes[2];
        args[0].v_int64 = 0; // output index = 0
        tcodes[0] = kTVMArgInt;
        args[1].v_handle = &output;
        tcodes[1] = kTVMDLTensorHandle;
        TVMValue ret_val2;
        int ret_tcode2;
        CHECK(TVMFuncCall(get_output, args, tcodes, 2, &ret_val2, &ret_tcode2));

        printf("输出: [");
        for (int i = 0; i < 10; i++)
            printf("%s%.4f", i ? ", " : "", output_buf[i]);
        printf("]\n");
    }

    // ---------- 5. 清理 ----------
    CHECK(TVMFuncFree(get_output));
    CHECK(TVMFuncFree(run_func));
    CHECK(TVMFuncFree(set_input));
    CHECK(TVMModFree(gmod));
    CHECK(TVMFuncFree(factory));
    CHECK(TVMModFree(mod));
    printf("推理完成\n");
    return 0;
}