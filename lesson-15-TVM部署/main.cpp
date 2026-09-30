// main.cpp
// Lesson 15: TVM 部署 —— 用 C++ 加载编译好的 libmlp3.so 并推理
//
// 对比 Lesson 08 的 ONNX Runtime C API:
//   ORT:  CreateSession(env, "model.onnx", ...) → Run(...)
//   TVM:  Module::LoadFromFile("libmlp3.so")     → GetFunction("set_input"/"run"/"get_output") → 调用
//
// 关键区别:
//   ORT 的权重在 .onnx 里,每次加载时解析
//   TVM 的权重已编译进 .so（export_library 时嵌入）,加载即就绪
//
// 编译: 见 Makefile
// 运行: ./main

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>

#include <cstdio>
#include <cstdint>

// DLTensor 里的 dtype 编码: code=2 表示 float, bits=32, lanes=1
static const DLDataType kDLFloat32 = {kDLFloat, 32, 1};

// ---------- 辅助: 创建 DLTensor 包装一段已有的 float 内存 ----------
// TVM 的 set_input/get_output 用 DLTensor 传数据（类似 ORT 的 OrtValue）
// data/data_shape 由调用者管理生命周期, 不涉及 TVM 的内存分配
static DLTensor make_tensor(void *data, int64_t *shape, int ndim)
{
    DLTensor t;
    t.data = data;
    t.device = {kDLCPU, 0};
    t.ndim = ndim;
    t.shape = shape;
    t.strides = nullptr;
    t.byte_offset = 0;
    t.dtype = kDLFloat32;
    return t;
}

int main()
{
    // ---------- 1. 加载编译好的模型 ----------
    // 对比 ORT: CreateSession(env, "model.onnx", ...)
    tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("./libmlp3.so");
    std::printf("加载 libmlp3.so 成功\n");

    // 2. 用默认设备创建图执行器
    tvm::Device dev{kDLCPU, 0}; // CPU 设备

    // 3. 拿执行函数  (lib["default"] 创建 graph executor)
    //    TVM 的图执行器把"喂输入/跑/拿输出"拆成 3 个可查询的 PackedFunc
    //    对比 ORT: 所有操作都在 Run() 一个调用里完成
    tvm::runtime::PackedFunc factory = mod.GetFunction("default");
    tvm::runtime::Module gmod = factory(dev);
    auto set_input = gmod.GetFunction("set_input");
    auto run_func = gmod.GetFunction("run");
    auto get_output = gmod.GetFunction("get_output");

    // 4. 准备输入: X[1,256], 全 0.5（和 Python 端一致）
    float input_buf[1 * 256];
    for (int i = 0; i < 256; i++)
        input_buf[i] = 0.5f;

    // 5. set_input + run   (类似 m.set_input("X", ...) + m.run())
    int64_t input_shape[2] = {1, 256};
    DLTensor input = make_tensor(input_buf, input_shape, 2);
    set_input("X", &input);
    run_func();

    // 6. get_output + 验证
    float output_buf[1 * 10] = {}; // 输出: Y[1,10]
    int64_t output_shape[2] = {1, 10};
    DLTensor output = make_tensor(output_buf, output_shape, 2);
    get_output(0, &output);

    std::printf("输出: [");
    for (int i = 0; i < 10; i++)
        std::printf("%s%.4f", i ? ", " : "", output_buf[i]);
    std::printf("]\n");

    std::printf("推理完成\n");
    return 0;
}