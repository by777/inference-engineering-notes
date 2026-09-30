// C2_run.cpp
// Lesson 22 实验C: 对比 float 基准 vs int8 量化模型的输出
//
// 回答一个问题: MNN 官方量化工具量化出的 int8 模型, 推理输出和 float 差多少?
// 方法: 用同一份输入, 分别跑 dwconv_float.mnn 和 dwconv_int8.mnn, 逐位对比。
//
// ⚠️ 最大的坑: int8 模型的输出 tensor 不能直接读 host 指针!
//    - 直接读 host<float>()    → 读到的是 int8 字节被当成 float 解释 (垃圾值)
//    - 直接读 host<int8_t>()   → 读到的是打包后的 int8 值, 还没反量化, 也格式不对
//    正确做法: 用 copyToHostTensor 拷贝到新建的 host tensor, MNN 会自动做:
//        ① NC4HW4 → NCHW 布局转换
//        ② int8 → float 反量化 (乘 scaleOut)
//    (官方 demo/exec/pictureRecognition.cpp 第 117 行就是这么写的)
//
// 编译: bash C2_build.sh

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace MNN;

// 加载模型 + 跑一次推理, 返回输出 (float 向量)
std::vector<float> runModel(const char *modelPath, const std::vector<float> &inputData)
{
    auto interpreter = Interpreter::createFromFile(modelPath);
    if (!interpreter)
    {
        std::cerr << "加载失败: " << modelPath << std::endl;
        return {};
    }

    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU; // 用 CPU 后端
    auto session = interpreter->createSession(config);

    // 写输入
    auto inputTensor = interpreter->getSessionInput(session, "input");
    auto nchwTensor = Tensor::create<float>({1, 3, 8, 8}, (void *)inputData.data(), Tensor::CAFFE);
    inputTensor->copyFromHostTensor(nchwTensor);

    // 推理
    interpreter->runSession(session);

    // ★ 关键: 用官方推荐的方式读输出 (见文件头注释)
    auto outputTensor = interpreter->getSessionOutput(session, "output");
    auto dimType = outputTensor->getDimensionType(); // 输出原始布局类型
    if (outputTensor->getType().code != halide_type_float)
    {                                 // 如果输出不是 float (int8 模型就是)
        dimType = Tensor::TENSORFLOW; //   改用 TENSORFLOW 布局 (NHWC, 可被 copyToHostTensor 处理)
    }
    std::shared_ptr<Tensor> outHost(new Tensor(outputTensor, dimType)); // 新建 host tensor
    outputTensor->copyToHostTensor(outHost.get());                      // ★ 自动布局转换 + 反量化

    auto outPtr = outHost->host<float>(); // 现在才是真正的 float 输出
    int n = outHost->elementSize();
    std::vector<float> result(outPtr, outPtr + n);

    interpreter->releaseSession(session);
    delete interpreter;
    return result;
}

int main()
{
    // ── ① 构造同一份输入 (和实验A一样, 范围 [-0.08, 0.08]) ──
    std::vector<float> inputData(1 * 3 * 8 * 8);
    for (int i = 0; i < (int)inputData.size(); i++)
    {
        inputData[i] = (i % 17) * 0.01f - 0.08f;
    }

    // ── ② 跑 float 基准 ──
    auto outFloat = runModel("dwconv_float.mnn", inputData);

    // ── ③ 跑 int8 量化模型 (quantized.out 量化) ──
    auto outInt8 = runModel("dwconv_int8.mnn", inputData);

    if (outFloat.empty() || outInt8.empty())
    {
        std::cerr << "推理失败" << std::endl;
        return 1;
    }

    // ── ④ 对比 ──
    std::cout << "=== float vs int8 输出对比 (前 12 个值) ===" << std::endl;
    float maxAbsDiff = 0.0f;
    int maxIdx = -1;
    for (int i = 0; i < (int)outFloat.size(); i++)
    {
        float diff = std::abs(outFloat[i] - outInt8[i]);
        if (diff > maxAbsDiff)
        {
            maxAbsDiff = diff;
            maxIdx = i;
        }
        if (i < 12)
        {
            std::cout << "  [" << i << "] float=" << outFloat[i]
                      << "  int8=" << outInt8[i]
                      << "  diff=" << diff << std::endl;
        }
    }
    std::cout << "... (共 " << outFloat.size() << " 个值)" << std::endl;
    std::cout << "最大绝对误差 = " << maxAbsDiff << " (下标 " << maxIdx << ")" << std::endl;
    std::cout << "参考: ORT 标准 QDQ 误差 ≈ 0.0156; MNN KL 校准略粗" << std::endl;
    return 0;
}