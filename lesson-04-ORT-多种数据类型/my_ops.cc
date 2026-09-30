// my_ops.cc
// Lesson 04: 多种数据类型的自定义算子
//
// 核心技巧：用 C++ template 实现同一个算子支持多种类型
// 编译器会自动为每种类型生成不同的代码

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <onnxruntime_lite_custom_op.h>

// ======== Template 版本的算子 ========
// 支持: float32, float64, int32, int64

template <typename T>
struct MyAddScale
{
    float scale_;

    MyAddScale(const OrtApi *ort_api, const OrtKernelInfo *info)
    {
        Ort::ConstKernelInfo kernel_info(info);
        try
        {
            scale_ = kernel_info.GetAttribute<float>("scale");
        }
        catch (...)
        {
            scale_ = 1.0f;
        }
    }

    void Compute(const Ort::Custom::Tensor<T> &X,
                 const Ort::Custom::Tensor<T> &Y,
                 Ort::Custom::Tensor<T> &Z)
    {
        auto shape = X.Shape();
        auto *x_raw = X.Data();
        auto *y_raw = Y.Data();
        auto *z_raw = Z.Allocate(shape);

        // z = (x + y) * scale_
        for (int64_t i = 0; i < Z.NumberOfElement(); ++i)
        {
            z_raw[i] = (x_raw[i] + y_raw[i]) * static_cast<T>(scale_);
        }
    }

    static Ort::Status
    InferOutputShape(Ort::ShapeInferContext &ctx)
    {
        ctx.SetOutputShape(0, ctx.GetInputShape(0));
        return Ort::Status{nullptr};
    }
};

// ====== 注册函数 ======

extern "C" OrtStatus *ORT_API_CALL
RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base)
{
    try
    {
        Ort::InitApi(api_base->GetApi(ORT_API_VERSION));

        // 先创建所有版本（不在 scope 内）
        static auto my_add_float = Ort::Custom::CreateLiteCustomOp<MyAddScale<float>>(
            "MyAddFloat", "CPUExecutionProvider");

        static auto my_add_double = Ort::Custom::CreateLiteCustomOp<MyAddScale<double>>(
            "MyAddDouble", "CPUExecutionProvider");

        static auto my_add_int32 = Ort::Custom::CreateLiteCustomOp<MyAddScale<int32_t>>(
            "MyAddInt32", "CPUExecutionProvider");

        static auto my_add_int64 = Ort::Custom::CreateLiteCustomOp<MyAddScale<int64_t>>(
            "MyAddInt64", "CPUExecutionProvider");

        // 然后加到域里
        static Ort::CustomOpDomain domain{"my_domain"};
        domain.Add(my_add_float);
        domain.Add(my_add_double);
        domain.Add(my_add_int32);
        domain.Add(my_add_int64);

        Ort::UnownedSessionOptions session_options{options};
        session_options.Add(domain);

        return Ort::Status{nullptr};
    }
    catch (const Ort::Exception &e)
    {
        return Ort::Status{e}.release();
    }
    catch (const std::exception &e)
    {
        return Ort::Status{e}.release();
    }
}