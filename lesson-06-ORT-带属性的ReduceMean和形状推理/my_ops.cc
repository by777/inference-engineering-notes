// my_ops.cc
// Lesson 06: 带属性的 ReduceMean
//
// 输入:  X (2D float)
// 属性:  axis, keepdims
// 输出:  Y
//
// 约束:
//   - 只支持 2D 输入
//   - 只支持 axis = 0 或 1
//   - 只支持 float

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <onnxruntime_lite_custom_op.h>
#include <vector>
#include <stdexcept>

struct MyReduceMean
{
    int64_t axis_;
    int64_t keepdims_;
    MyReduceMean(const OrtApi *ort_api, const OrtKernelInfo *info)
    {
        Ort::ConstKernelInfo kernel_info(info);
        try
        {
            axis_ = kernel_info.GetAttribute<int64_t>("axis");
        }
        catch (...)
        {
            axis_ = 1;
        }
        try
        {
            keepdims_ = kernel_info.GetAttribute<int64_t>("keepdims");
        }
        catch (...)
        {
            keepdims_ = 0;
        }
        if (axis_ != 0 && axis_ != 1)
        {
            throw std::runtime_error("MyReduceMean only supports axis=0 or axis=1");
        }

        if (keepdims_ != 0 && keepdims_ != 1)
        {
            throw std::runtime_error("MyReduceMean only supports keepdims=0 or keepdims=1");
        }
    }

    void Compute(const Ort::Custom::Tensor<float> &X, Ort::Custom::Tensor<float> &Y)
    {
        auto in_shape = X.Shape();
        if (in_shape.size() != 2)
        {
            throw std::runtime_error("MyReduceMean only supports 2D input");
        }
        const int64_t rows = in_shape[0];
        const int64_t cols = in_shape[1];

        const float *x_raw = X.Data();
        std::vector<int64_t> out_shape;
        if (axis_ == 0)
        {
            if (keepdims_ == 1)
            {
                out_shape = {1, cols};
            }
            else
            {
                out_shape = {cols};
            }
        }
        else
        {
            if (keepdims_ == 1)
            {
                out_shape = {rows, 1};
            }
            else
            {
                out_shape = {rows};
            }
        }
        float *y_raw = Y.Allocate(out_shape);
        if (axis_ == 0)
        {
            for (int64_t c = 0; c < cols; ++c)
            {
                float sum = 0.0f;
                for (int64_t r = 0; r < rows; ++r)
                {
                    sum += x_raw[r * cols + c];
                }
                y_raw[c] = sum / static_cast<float>(rows);
            }
        }
        else
        {
            for (int64_t r = 0; r < rows; ++r)
            {
                float sum = 0.0f;
                for (int64_t c = 0; c < cols; ++c)
                {
                    sum += x_raw[r * cols + c];
                }
                y_raw[r] = sum / static_cast<float>(cols);
            }
        }
    }

    static Ort::Status InferOutputShape(Ort::ShapeInferContext &ctx)
    {
        auto in_shape = ctx.GetInputShape(0);

        if (in_shape.size() != 2)
        {
            return Ort::Status{"MyReduceMean only supports 2D input", ORT_INVALID_ARGUMENT};
        }

        int64_t axis = 1;
        int64_t keepdims = 0;

        try
        {
            axis = ctx.GetAttrInt("axis");
        }
        catch (...)
        {
            axis = 1;
        }

        try
        {
            keepdims = ctx.GetAttrInt("keepdims");
        }
        catch (...)
        {
            keepdims = 0;
        }

        if (axis != 0 && axis != 1)
        {
            return Ort::Status{"MyReduceMean only supports axis=0 or axis=1", ORT_INVALID_ARGUMENT};
        }

        if (keepdims != 0 && keepdims != 1)
        {
            return Ort::Status{"MyReduceMean only supports keepdims=0 or keepdims=1", ORT_INVALID_ARGUMENT};
        }

        Ort::ShapeInferContext::Shape out_shape;
        if (axis == 0)
        {
            if (keepdims == 1)
            {
                out_shape.emplace_back(1);
                out_shape.push_back(in_shape[1]);
            }
            else
            {
                out_shape.push_back(in_shape[1]);
            }
        }
        else
        {
            if (keepdims == 1)
            {
                out_shape.push_back(in_shape[0]);
                out_shape.emplace_back(1);
            }
            else
            {
                out_shape.push_back(in_shape[0]);
            }
        }

        ctx.SetOutputShape(0, out_shape);
        return Ort::Status{nullptr};
    }
};

extern "C" OrtStatus *ORT_API_CALL
RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base)
{
    try
    {
        Ort::InitApi(api_base->GetApi(ORT_API_VERSION));

        static auto my_reduce_mean_op =
            Ort::Custom::CreateLiteCustomOp<MyReduceMean>(
                "MyReduceMean",
                "CPUExecutionProvider");

        static Ort::CustomOpDomain domain{"my_domain"};
        domain.Add(my_reduce_mean_op);

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