// 03_mnn_cpp.cpp
// Lesson 19: MNN 初识 - 实验 3: C++ 推理（对比 TVM L15/16 部署）
//
// 对比:
//   TVM L15/16: Interpreter → set_input/run/get_output（graph_executor）
//   MNN C++:    Interpreter → createSession → 输入拷贝 → runSession → 输出拷贝
//
// 编译: g++ 03_mnn_cpp.cpp -o mnn_cpp -I../mnn-src/include -L../mnn-src/build -lMNN -std=c++11
// 运行: ./mnn_cpp

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>

using namespace MNN;

int main()
{
    // ========== 1. 加载 .mnn 模型 ==========
    // 类比 TVM: TVMModLoadFromFile（加载模型）
    // 类比 ORT: InferenceSession
    std::shared_ptr<Interpreter> interpreter(Interpreter::createFromFile("mlp3.mnn"));
    if (!interpreter)
    {
        fprintf(stderr, "加载 mlp3.mnn 失败\n");
        return 1;
    }
    printf("加载 mlp3.mnn 成功\n");

    // ========== 2. 创建推理会话 ==========
    // createSession: 运行时做内存分配 + 图优化（类比 TVM 的 default 工厂）
    // 注意: MNN 的调度（session）配置可以指定后端（CPU/GPU），这里用默认 CPU
    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU; // CPU 推理
    // ⚠️ MNN 3.6.1 API: createSession 返回 Session*（旧版返回 int）
    Session* session = interpreter->createSession(config);
    printf("创建会话成功, session=%p\n", (void*)session);

    // ========== 3. 拿输入/输出张量 ==========
    // getSessionInput/Output: 和 Python 的 getSessionInput 对应
    auto input_tensor = interpreter->getSessionInput(session, "X");
    auto output_tensor = interpreter->getSessionOutput(session, "Y");
    printf("输入: %d 维, 输出: %d 维\n",
           input_tensor->dimensions(), output_tensor->dimensions());

    // ========== 4. 准备输入数据 ==========
    // 和 Python 一样: 需要"宿主张量"（host tensor），然后 copyFromHostTensor
    const int input_size = 1 * 256;
    std::vector<float> input_data(input_size);
    for (int i = 0; i < input_size; i++)
        input_data[i] = 0.5f; // 简单填 0.5（和实验 2 数据不同，方便看输出差异）

    // 创建宿主张量（形状、类型、数据）
    // ⚠️ MNN 3.6.1 API: 用 Tensor::create（不是 createTensor）
    //    且布局参数默认 TENSORFLOW，必须显式传 CAFFE（NCHW）
    std::vector<int> input_shape = {1, 256};
    auto host_input = Tensor::create(
        input_shape,             // 形状 [1, 256]
        halide_type_of<float>(), // float32
        input_data.data(),       // 数据指针
        Tensor::CAFFE            // NCHW 布局（对应实验 2 的 Caffe 枚举）
    );
    // 拷贝: 宿主数据 → MNN 内部张量（内部是 NC4HW4 优化布局）
    // ⚠️ Tensor::create 返回裸指针 Tensor*（不是 shared_ptr），直接用
    input_tensor->copyFromHostTensor(host_input);

    // ========== 5. 跑推理 ==========
    interpreter->runSession(session);
    printf("推理完成\n");

    // ========== 6. 取输出 ==========
    // 创建宿主输出张量，copyToHostTensor 把结果搬到 numpy 可读的形式
    auto output_shape = output_tensor->shape();
    std::vector<float> output_data(1 * 10);
    auto host_output = Tensor::create(
        output_shape, halide_type_of<float>(), output_data.data(), Tensor::CAFFE);
    output_tensor->copyToHostTensor(host_output);

    // 打印结果
    printf("输出 (前 10 个): [");
    for (int i = 0; i < 10; i++)
        printf("%s%.4f", i ? ", " : "", output_data[i]);
    printf("]\n");

    return 0;
}