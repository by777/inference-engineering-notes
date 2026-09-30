#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_lite_custom_op.h>

void KernelMyAdd(const Ort::Custom::Tensor<float> &X,
                 const Ort::Custom::Tensor<float> &Y,
                 Ort::Custom::Tensor<float> &Z)
{
    auto shape = X.Shape();
    auto *x_raw = X.Data();
    auto *y_raw = Y.Data();
    auto *z_raw = Z.Allocate(shape);
    for (int64_t i = 0; i < Z.NumberOfElement(); i++)
    {
        z_raw[i] = x_raw[i] + y_raw[i];
    }
}

int main()
{
    std::cout << "Hello, ONNX Runtime Custom Operator!" << std::endl;

    // 第2部分：注册算子把函数指针 KernelMyAdd 包装成一个 OrtCustomOp 结构体（C ABI 结构）
    // 三个参数必须和模型对上：算子名、执行提供者、内核函数指针
    std::unique_ptr<Ort::Custom::OrtLiteCustomOp> my_add_op{
        Ort::Custom::CreateLiteCustomOp("MyAdd", "CPUExecutionProvider", KernelMyAdd)};
    // 把算子放进一个"域容器"。"my_domain" 必须等于 gen_model.py 里 make_node(..., domain="my_domain") 和 make_opsetid("my_domain", 1) 的字符串
    Ort::CustomOpDomain domain{"my_domain"};
    domain.Add(my_add_op.get());
    // 把整个域注册到 SessionOptions
    Ort::SessionOptions session_options;
    session_options.Add(domain);

    // 第3部分：加载模型
    // Env 是 ORT 全局环境，一个进程一个就够，负责日志和线程池
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "step1");
    const char *model_path = "my_add.onnx";
    // Session 构造时做这几件事：
    // 解析 ONNX 文件
    // 读 opset_imports → 建算子查找表
    // 遇到 MyAdd(domain=my_domain) → 去 session_options 上的自定义域找 → 找到 my_add_op → 注册 kernel
    // 编译执行图
    Ort::Session session{env, model_path, session_options};

    // 第4部分：准备输入数据
    std::vector<float> x_data{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y_data{10.0f, 20.0f, 30.0f, 40.0f};
    std::array<int64_t, 1> shape{4};

    // 创建内存信息
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_x = Ort::Value::CreateTensor<float>(
        memory_info, x_data.data(), x_data.size(), shape.data(), shape.size());
    Ort::Value input_y = Ort::Value::CreateTensor<float>(
        memory_info, y_data.data(), y_data.size(), shape.data(), shape.size());

    // 第5部分：跑推理
    const char *input_names[] = {"X", "Y"};
    const char *output_names[] = {"Z"};
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_x, 2,
        output_names, 1);
    float *z = outputs[0].GetTensorMutableData<float>();
    for (size_t i = 0; i < 4; ++i)
        std::cout << z[i] << " ";
    std::cout << "\n  期望:       11 22 33 44   ";
    bool ok = (z[0] == 11.f && z[1] == 22.f && z[2] == 33.f && z[3] == 44.f);
    std::cout << (ok ? "✅" : "❌") << "\n";
    return ok ? 0 : 1;
    return 0;
}