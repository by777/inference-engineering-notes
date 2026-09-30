// my_ops.cc
// Lesson 02: 把 MyAdd 算子打包成可动态加载的 .so
//
// 核心区别 vs Lesson 01:
//   - 不写 main(), 不创建 session, 只导出算子
//   - 必须导出 RegisterCustomOps 符号, ORT 运行时会调它
//   - 用 ORT_API_MANUAL_INIT 模式: 因为 .so 不链接 ORT 主程序, 需要手动初始化 API 表
//
// 编译:
//   g++ my_ops.cc -o libmy_ops.so -shared -fPIC \
//       -I../ort-bin/include -std=c++17
// 注意: 不需要 -lonnxruntime! 算子库不直接链接 ORT
//       它通过 ORT 传进来的 OrtApiBase* 拿到 API 表

#define ORT_API_MANUAL_INIT      // 影响接下来 include 的头文件
#include <onnxruntime_cxx_api.h> // 头文件内 #ifdef 选 MANUAL_INIT 分支, 关掉自动初始化
#undef ORT_API_MANUAL_INIT       // 取消宏, 不污染后续代码

#include <onnxruntime_lite_custom_op.h>

// ====== 算子计算函数 (和第1课完全一样) ======
void KernelMyAdd(const Ort::Custom::Tensor<float> &X,
                 const Ort::Custom::Tensor<float> &Y,
                 Ort::Custom::Tensor<float> &Z)
{
    auto shape = X.Shape();
    auto *x_raw = X.Data();
    auto *y_raw = Y.Data();
    auto *z_raw = Z.Allocate(shape);
    for (int64_t i = 0; i < Z.NumberOfElement(); ++i)
    {
        z_raw[i] = x_raw[i] + y_raw[i];
    }
}

// ====== 算子注册函数 ======
// ORT 加载 .so 时通过 dlsym 找到 RegisterCustomOps 符号并调用
// 参数:
//   options  - ORT 传进来的 SessionOptions, 我们往里加 domain
//   api_base - ORT 的 API 基类指针, 用来拿到完整 API 表
//
// 必须用 C 链接 (extern "C") 导出, 否则 C++ 会 name mangle 符号名
extern "C" OrtStatus *ORT_API_CALL
// extern "C"   —禁止 C++ name mangling
// OrtStatus *  — 返回类型
// ORT_API_CALL —调用约定（calling convention）规定函数参数怎么入栈、谁清栈。Windows 上 MSVC 默认 __cdecl，但 ORT 的 C API 约定用 __stdcall，两边必须一致才能正确调用。Linux 上没有调用约定区分，所以宏为空。
RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base)
{

    if (!api_base)
    {
        return Ort::Status("api_base is nullptr", ORT_FAIL).release();
    }
    const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
    if (!api)
    {
        return Ort::Status("GetApi failed", ORT_FAIL).release();
    }
    Ort::InitApi(api);
    OrtStatus *status = nullptr;
    try
    {
        // 关键：op和domain都必须是static
        // 否则函数返回后析构，ORT内部持有的指针变悬空，session创建时段错误
        static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> my_add_op{
            Ort::Custom::CreateLiteCustomOp("MyAdd", "CPUExecutionProvider", KernelMyAdd)};
        static Ort::CustomOpDomain domain{"my_domain"};
        domain.Add(my_add_op.get());
        Ort::UnownedSessionOptions session_options{options};
        session_options.Add(domain);
    }
    catch (const Ort::Exception &e)
    {
        Ort::Status ort_status{e};
        status = ort_status.release();
    }
    catch (const std::exception &e)
    {
        Ort::Status ort_status{e};
        status = ort_status.release();
    }
    catch (...)
    {
        Ort::Status ort_status{"unknown exception", ORT_FAIL};
        status = ort_status.release();
    }
    return status;
}