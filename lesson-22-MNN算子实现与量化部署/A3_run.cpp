// A3_run.cpp
// Lesson 22 实验A: 用 MNN C++ API 跑 depthwise conv 模型
// 编译: g++ -std=c++11 A3_run.cpp -I../mnn-src/include -L../mnn-src/build -lMNN -o A3_run

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace MNN;

int main(int argc, char *argv[])
{
    const char *modelPath = "dwconv.mnn";
    if (argc > 1)
        modelPath = argv[1];

    // ① 加载模型
    auto interpreter = Interpreter::createFromFile(modelPath);
    if (!interpreter)
    {
        std::cerr << "加载失败" << std::endl;
        return 1;
    }
    std::cout << "[0] 模型加载成功" << std::endl;

    // ② 创建会话（CPU 后端）
    ScheduleConfig sConfig;
    sConfig.type = MNN_FORWARD_CPU;
    auto session = interpreter->createSession(sConfig);
    std::cout << "[1] 会话创建成功 (CPU 后端)" << std::endl;

    // ③ 获取输入
    auto inputTensor = interpreter->getSessionInput(session, "input");
    std::cout << "[2] 输入 tensor shape=";
    for (auto d : inputTensor->shape())
        std::cout << d << ",";
    std::cout << std::endl;

    // ④ 写入输入数据
    std::vector<float> inputData(1 * 3 * 8 * 8);
    for (int i = 0; i < inputData.size(); i++)
        inputData[i] = (i % 17) * 0.01f - 0.08f; // 有规律的测试数据
    auto nchwTensor = Tensor::create<float>({1, 3, 8, 8}, inputData.data(), Tensor::CAFFE);
    inputTensor->copyFromHostTensor(nchwTensor);

    // ⑤ 推理
    auto start = std::chrono::high_resolution_clock::now();
    interpreter->runSession(session);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "[3] 推理完成, 耗时 " << ms << " ms" << std::endl;

    // ⑥ 取输出
    auto outputTensor = interpreter->getSessionOutput(session, "output");
    auto outPtr = outputTensor->host<float>();
    std::cout << "[4] 输出前 16 个值: ";
    for (int i = 0; i < 16; i++)
        std::cout << outPtr[i] << " ";
    std::cout << std::endl;

    interpreter->releaseSession(session);
    delete interpreter;
    return 0;
}