// C3_trace.cpp
// Lesson 22 实验C: 观察 MNN int8 模型的运行时执行链
//
// 回答一个问题: int8 模型运行时到底执行哪些算子?
//   静态看 dwconv_int8.json: 只有 ConvertTensor + ConvolutionDepthwise + ConvertTensor,
//   没有 FloatToInt8/Int8ToFloat 节点!
//   但运行时呢? 用回调 API 打印每个算子的 type, 真相大白:
//       FloatToInt8 [ DT_INT8 ] → Raster → ConvolutionDepthwise [ DT_INT8 ] → Raster
//   结论: MNN 调度器根据 tensor 的 quantAttr, 运行时自动插入量化转换算子!
//   这就是"自动 int8 通路" —— 用户无感, 模型侧也不写死, 调度时动态决定。
//
// 算子名带 " [ DT_INT8 ] " 后缀 = 该算子输出是 int8 类型 (Pipeline.cpp 动态拼的)
//
// ⚠️ 局限: 回调拿到的 tensor 是 MNN 内部格式 (NC4HW4 + 打包), 直接读
//    host<float>() / host<int8_t>() 读到的是打包字节不是真实值。
//    (要读真实数值, 用 C2_run.cpp 的 copyToHostTensor 方法)
//    但本程序只看算子的 type/name —— 这些完全可信。
//
// 编译: bash C3_build.sh

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <iostream>
#include <vector>

using namespace MNN;

int main()
{
    auto interpreter = Interpreter::createFromFile("dwconv_int8.mnn");
    if (!interpreter)
    {
        std::cerr << "加载失败" << std::endl;
        return 1;
    }

    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    auto session = interpreter->createSession(config);

    // 写输入 (和 C2 同一份)
    auto inputTensor = interpreter->getSessionInput(session, "input");
    std::vector<float> inputData(1 * 3 * 8 * 8);
    for (int i = 0; i < (int)inputData.size(); i++)
        inputData[i] = (i % 17) * 0.01f - 0.08f;
    auto nchw = Tensor::create<float>({1, 3, 8, 8}, inputData.data(), Tensor::CAFFE);
    inputTensor->copyFromHostTensor(nchw);

    // before 回调: 每层执行前打印算子类型和名字
    // info->type(): 算子类型字符串, 如 "FloatToInt8 [ DT_INT8 ]"
    // info->name(): 算子名
    TensorCallBackWithInfo before = [](const std::vector<Tensor *> &tensors, const OperatorInfo *info)
    {
        std::cout << "[before] " << info->type() << "  name=" << info->name() << std::endl;
        return true; // true = 继续执行该层
    };
    TensorCallBackWithInfo after = [](const std::vector<Tensor *> &tensors, const OperatorInfo *info)
    {
        std::cout << "[after ] " << info->type() << "  name=" << info->name() << std::endl;
        return true;
    };

    std::cout << "=== int8 模型逐层执行链 ===" << std::endl;
    interpreter->runSessionWithCallBackInfo(session, before, after);

    interpreter->releaseSession(session);
    delete interpreter;
    return 0;
}