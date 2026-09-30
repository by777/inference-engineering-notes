// my_ops.cc
// Lesson 05: 多输入多输出自定义算子
//
// 输入:
//   X, Y
// 输出:
//   Sum  = X + Y
//   Diff = X - Y

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <onnxruntime_lite_custom_op.h>

struct MyAddSub
{
    MyAddSub(const OrtApi *ort_api, const OrtKernelInfo *info)
    {
        // 这里没有参数要读, 所以构造函数空着
    }
    void Compute(const Ort::Custom::Tensor<float> &X,
                 const Ort::Custom::Tensor<float> &Y,
                 Ort::Custom::Tensor<float> &Sum,
                 Ort::Custom::Tensor<float> &Diff)
    {
        auto shape = X.Shape();
        auto *x_raw = X.Data();
        auto *y_raw = Y.Data();
        auto *sum_raw = Sum.Allocate(shape);
        auto *diff_raw = Diff.Allocate(shape);
        for (int64_t i = 0; i < Sum.NumberOfElement(); ++i)
        {
            sum_raw[i] = x_raw[i] + y_raw[i];
            diff_raw[i] = x_raw[i] - y_raw[i];
        }
    }
    static Ort::Status InferOutputShape(Ort::ShapeInferContext &ctx)
    {
        ctx.SetOutputShape(0, ctx.GetInputShape(0));
        ctx.SetOutputShape(1, ctx.GetInputShape(0));
        return Ort::Status{nullptr};
    }
};

extern "C" OrtStatus *ORT_API_CALL
RegisterCustomOps(OrtSessionOptions *options,
                  const OrtApiBase *api_base)
{
    try
    {
        Ort::InitApi(api_base->GetApi(ORT_API_VERSION));

        static auto my_add_sub_op =
            Ort::Custom::CreateLiteCustomOp<MyAddSub>(
                "MyAddSub",
                "CPUExecutionProvider");

        static Ort::CustomOpDomain domain{"my_domain"};
        domain.Add(my_add_sub_op);

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