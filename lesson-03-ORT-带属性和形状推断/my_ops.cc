// my_ops.cc
// Lesson 03: 带属性和 Shape Inference 的自定义算子
//
// 改进点 vs Lesson 02:
//   - 用 struct 封装算子, 从 KernelInfo 读属性
//   - 加 InferOutputShape 静态方法做 shape 推断
//   - 输出 = (X + Y) * scale, scale 从模型属性读

#define ORT_API_MANUAL_INIT      // 影响接下来 include 的头文件
#include <onnxruntime_cxx_api.h> // 头文件内 #ifdef 选 MANUAL_INIT 分支, 关掉自动初始化
#undef ORT_API_MANUAL_INIT       // 取消宏, 不污染后续代码

#include <onnxruntime_lite_custom_op.h>

// ====== MyAdd 算子，用struct封装 ======
// struct的好处是： 构造函数可以读取KernelInfo里的属性
// 对比函数板子，属性必须通过其他机制传，比较麻烦
/*
┌─────────────────┬──────────────────────┬──────────────────────┐
│ 维度            │ 函数版               │ Struct版             │
├─────────────────┼──────────────────────┼──────────────────────┤
│ 参数来源        │ 只有 Tensor          │ Tensor + KernelInfo  │
│ 属性读取        │ 无法读               │ 构造函数里读         │
│ 参数存储        │ 硬编码 or 其他机制   │ 成员变量             │
│ 方法形式        │ 全局函数指针         │ 实例方法（Compute）  │
│ ORT 调用方式    │ 每次直接调函数       │ 先构造，再调Compute  │
│ 属性改变后      │ 需要重新编译         │ 构造函数自动读新值   │
│ 数据驱动参数    │ ✗ 不支持             │ ✓ 支持               │
└─────────────────┴──────────────────────┴──────────────────────┘
*/

// ============================================================
// 版本对比：函数版 vs Struct版
// ============================================================

//     // ❌ 问题 1: 参数固定，只能是 Tensor
//     //    没办法访问属性（scale、axis 等）
//
//     // ❌ 问题 2: 如果需要参数，只能硬编码
//     //    比如这里 scale 写死了：
//     auto x_raw = X.DataAsSpan();
//     auto y_raw = Y.DataAsSpan();
//     auto z_raw = Z.DataAsSpan();
//
//     float scale = 2.5f;  // ← 硬编码！改模型属性也没用
//
//     for (size_t i = 0; i < z_raw.size(); ++i) {
//         z_raw[i] = (x_raw[i] + y_raw[i]) * scale;
//     }
//     // ❌ 问题 3: 无法利用 KernelInfo 里的其他信息

// ─────────────────────────────────────────────────────────
// ✅ Struct版（Lesson 03 的方式）
// ─────────────────────────────────────────────────────────

struct MyAddScale
{
    float scale_; // 成员变量:存属性
    // 构造函数:ORT创建Kernel时自动调用
    // 参数里有KernelInfo，可以从中读取属性
    MyAddScale(const OrtApi *ort_api, const OrtKernelInfo *info)
    {
        // 优势1: 可以访问KernelInfo
        Ort::ConstKernelInfo kernel_info(info);
        try
        {
            scale_ = kernel_info.GetAttribute<float>("scale");
        }
        catch (...)
        {
            scale_ = 1.0f; // 默认值
        }
    }
    void Compute(const Ort::Custom::Tensor<float> &X,
                 const Ort::Custom::Tensor<float> &Y,
                 Ort::Custom::Tensor<float> &Z)
    {
        auto shape = X.Shape();
        auto *x_raw = X.Data();
        auto *y_raw = Y.Data();
        auto *z_raw = Z.Allocate(shape);
        for (int64_t i = 0; i < Z.NumberOfElement(); ++i)
        {
            z_raw[i] = (x_raw[i] + y_raw[i]) * scale_;
        }
    }
    // Shape 推断（可选但推荐）
    //    告诉 ORT 输出 shape 是什么
    static Ort::Status
    InferOutputShape(Ort::ShapeInferContext &ctx)
    {
        // 这个算子不改 shape，输出和第一个输入一样
        ctx.SetOutputShape(0, ctx.GetInputShape(0));
        return Ort::Status{nullptr}; // 成功
    }
};
// ====== 注册函数 ======
// ORT 会通过 dlsym 查找这个函数，并调用它
extern "C" OrtStatus *ORT_API_CALL
RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base)
{
    try
    { // 初始化 ORT API（连接到 ORT Runtime）
        Ort::InitApi(api_base->GetApi(ORT_API_VERSION));

        // 创建 MyAddScale 算子的实例
        // <MyAddScale> 告诉 ORT：这个算子用 struct 版本，有构造函数 + Compute + InferOutputShape
        static std::unique_ptr<Ort::Custom::OrtLiteCustomOp> my_add_op{
            Ort::Custom::CreateLiteCustomOp<MyAddScale>(
                "MyAdd",                                //   算子名字（模型里用这个名字）和
                "CPUExecutionProvider")};               // 执行器（都用 CPU）
                                                        // 创建自定义算子域
        static Ort::CustomOpDomain domain{"my_domain"}; // 域名（模型里用这个域名）
        domain.Add(my_add_op.get());                    // 把 MyAdd 算子加到域里
        // 把域加到 Session 选项

        Ort::UnownedSessionOptions session_options{options};
        session_options.Add(domain);

        return Ort::Status{nullptr}; // 注册成功
    }
    catch (const Ort::Exception &e)
    {
        return Ort::Status{e}.release(); // 直接传 Exception 对象
    }
    catch (const std::exception &e)
    {
        return Ort::Status{e}.release(); // 直接传 std::exception 对象
    }
}