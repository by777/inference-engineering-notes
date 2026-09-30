// A4_trace.cpp
// Lesson 22 实验A: 用 MNN 官方回调 API 跟踪逐层调用链 (带算子类型)
//
// 这个程序回答一个问题: MNN 推理时, 模型里的算子到底按什么顺序执行?
// 方法: 在 MNN 的执行流程上挂两个钩子(回调), 每层执行前/后各触发一次,
//       打印出算子的类型和名字 → 得到完整的调用链。
//
// 核心 API: runSessionWithCallBackInfo(session, before, after)
//   - 它是 runSession 的"可观测版本", 在算子循环里故意留了回调点:
//       for (每个算子) {
//           before(输入tensors, 算子信息);   // ← 每层执行前
//           执行这个算子;
//           after(输入tensors, 算子信息);    // ← 每层执行后
//       }
//   - before/after 是 std::function 回调, 你在里面打印/统计/甚至返回 false 跳过某层
//
// 为什么用这个而不是改源码加日志?
//   - libMNN.so 是编译好的黑盒 (还 stripped 无符号), 改不了
//   - MNN 官方提供了回调钩子 = 白盒调试的正路, 零源码改动
//   - 这是"在公司不能改框架源码时怎么观察框架行为"的标准技能
//
// 编译: g++ -std=c++11 A4_trace.cpp -I../mnn-src/include -L../mnn-src/build -lMNN -o A4_trace
// 运行: LD_LIBRARY_PATH=../mnn-src/build ./A4_trace

#include <MNN/Interpreter.hpp> // MNN 主 API: Interpreter 类, 回调类型定义
#include <MNN/Tensor.hpp>      // MNN 张量: Tensor 类, 拿 shape/数据
#include <iostream>            // std::cout 打印
#include <vector>              // std::vector

using namespace MNN; // MNN 命名空间, 省去 MNN:: 前缀

int main(int argc, char *argv[])
{
    const char *modelPath = "dwconv.mnn";
    if (argc > 1)
        modelPath = argv[1]; // 支持 ./A4_trace 其它.mnn

    // ── ① 加载模型 (和 A3_run.cpp 一样, 略) ──
    auto interpreter = Interpreter::createFromFile(modelPath);
    if (!interpreter)
    {
        std::cerr << "加载失败" << std::endl;
        return 1;
    }
    std::cout << "[0] 模型加载成功" << std::endl;

    // ── ② 创建会话 (和 A3_run.cpp 一样) ──
    // ScheduleConfig: 会话配置, type=MNN_FORWARD_CPU 指定 CPU 后端
    ScheduleConfig sConfig;
    sConfig.type = MNN_FORWARD_CPU;
    auto session = interpreter->createSession(sConfig);

    // ── ③ 写入输入数据 (和 A3_run.cpp 一样) ──
    auto inputTensor = interpreter->getSessionInput(session, "input");
    std::vector<float> inputData(1 * 3 * 8 * 8);
    for (int i = 0; i < inputData.size(); i++)
        inputData[i] = (i % 17) * 0.01f - 0.08f; // 有规律的测试数据
    // Tensor::create: 创建 host 张量 (Tensor::CAFFE = NCHW 布局)
    // copyFromHostTensor: 把 host 数据拷进 session 的内部张量
    auto nchwTensor = Tensor::create<float>({1, 3, 8, 8}, inputData.data(), Tensor::CAFFE);
    inputTensor->copyFromHostTensor(nchwTensor);

    // ══════════ 关键部分: 定义两个回调 (钩子) ══════════

    // before 回调: 每层执行前被调
    // 参数1 tensors: 这层的输入 tensor 列表 (只读观察)
    // 参数2 info: OperatorInfo 指针, 拿算子类型和名字
    // 返回 true = 继续执行这层; false = 跳过这层 (钩子的威力, 能干预!)
    TensorCallBackWithInfo before = [](const std::vector<Tensor *> &tensors, const OperatorInfo *info)
    {
        // info->type(): 算子类型字符串, 如 "ConvolutionDepthwise"
        // info->name(): 算子名 (MNN 里常和输出 tensor 同名)
        std::cout << "  [before] type=" << info->type()
                  << "  name=" << info->name();
        // 顺带打印第一个输入 tensor 的形状, 方便对上模型结构
        if (!tensors.empty())
        {
            auto t = tensors[0];
            std::cout << "  in_shape=";
            for (auto d : t->shape()) // shape() 返回 vector<int>
                std::cout << d << ",";
        }
        std::cout << std::endl;
        return true; // 继续执行这层
    };

    // after 回调: 每层执行后被调 (这里只打印类型和名字)
    // lambda：[](参数) { 函数体 } 定义匿名函数，可以赋给变量、传参
    // TensorCallBackWithInfo 是类型别名（typedef std::function<...>），声明了一个"参数是 tensors+info、返回 bool"的函数签名
    // 回调 = 把函数当参数传：你把 before 这个函数传给 MNN，MNN 在特定时机（每层执行前）调用它。这是"钩子"的 C++ 实现
    TensorCallBackWithInfo after = [](const std::vector<Tensor *> &tensors, const OperatorInfo *info)
    {
        std::cout << "  [after]  type=" << info->type()
                  << "  name=" << info->name() << std::endl;
        return true;
    };

    std::cout << "[1] 开始推理 (逐层跟踪, 含算子类型):" << std::endl;
    // runSessionWithCallBackInfo: 带回调的推理, 代替 runSession
    interpreter->runSessionWithCallBackInfo(session, before, after);

    // ── ④ 取输出 (和 A3_run.cpp 一样) ──
    auto outputTensor = interpreter->getSessionOutput(session, "output");
    auto outPtr = outputTensor->host<float>();
    std::cout << "[2] 输出前 16 个值: ";
    for (int i = 0; i < 16; i++)
        std::cout << outPtr[i] << " ";
    std::cout << std::endl;

    interpreter->releaseSession(session); // 释放会话
    delete interpreter;                   // 释放解释器
    return 0;
}