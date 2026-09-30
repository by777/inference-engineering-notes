// main.c
// Lesson 09: Session 内省 —— 用 C API 查看模型的结构
// 内容: ORT 版本 / 可用 EP / 输入输出(名字、类型、shape、symbolic dim) / 可覆盖 initializer / model metadata
// 编译: gcc -std=c17 -I../ort-bin/include -L../ort-bin/lib -lonnxruntime -Wl,-rpath,../ort-bin/lib -o main main.c

#include <stdio.h>
#include <stdlib.h>
#include "onnxruntime_c_api.h"
/*
main()
 ├─ 0. 拿 api 入口 + 查 ORT 版本 / 可用 EP
 ├─ 1. 输入:     Count → Name → TypeInfo → print
 ├─ 2. 可覆盖权重: Count → Name → TypeInfo → print   (B 这类)
 ├─ 3. 输出:     Count → Name → TypeInfo → print
 ├─ 4. 元数据:   Producer/Graph名/Domain/描述/版本/自定义keys
 └─ 5. 逆序释放
*/
// ================================================================
// CHECK_STATUS: 错误检查宏
// ORT 的 C API 所有函数都返回 OrtStatus*:
//   - NULL  = 成功
//   - 非NULL = 失败，里面含有错误信息
// 这个宏把"调用函数 + 检查错误 + 打印 + 退出"压缩成一行。
// 注意: 宏体里用的 api 是外层 main 里的局部变量，靠宏的展开借用它。
// ================================================================
#define CHECK_STATUS(expr)                             \
    do                                                 \
    {                                                  \
        OrtStatus *s = (expr);                         \
        if (s != NULL)                                 \
        {                                              \
            const char *msg = api->GetErrorMessage(s); \
            fprintf(stderr, "Error: %s\n", msg);       \
            api->ReleaseStatus(s);                     \
            exit(1);                                   \
        }                                              \
    } while (0)

// ================================================================
// elem_type_str: 把枚举值翻译成人类可读的字符串
// ONNXTensorElementDataType 是 int 枚举，直接 printf("%d") 只能看到
// 数字 1、6、7... 没法读。这个函数做成一个查表翻译器。
// 为什么 FLOAT = 1 而 FLOAT16 = 10? 因为枚举值对齐了 ONNX 的
// TensorProto.DataType 编号，是历史遗留约定，所以必须查表。
// ================================================================
static const char *elem_type_str(ONNXTensorElementDataType t)
{
    switch (t)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return "uint32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return "uint64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return "string";
    default:
        return "other";
    }
}

// ================================================================
// print_tensor_info: 打印一个输入/输出的类型和形状
//
// 为什么需要两步 (TypeInfo -> TensorTypeAndShapeInfo)?
//   OrtValue 可以是 Tensor、Map、Sequence 等多种类型，所以 ORT 先用
//   OrtTypeInfo 这个"通用类型容器"描述它。
//   CastTypeInfoToTensorInfo 就是问它: "你里面是 Tensor 吗?"
//      - 是 Tensor: 返回指针，我们继续读 shape
//      - 不是: 返回 NULL（不会报错，所以要自己判断）
//
// shape 里有两个东西要看:
//   1. GetDimensions          -> 数值维度，比如 {1, 4}
//   2. GetSymbolicDimensions  -> 符号维度名，比如 "N"、"batch"
//   模型里写的是 [1, N]，运行时 N 会被 ORT 用 -1 占位（表示动态）。
//   所以我们同时拿两份，优先显示符号名，这样 [1, -1] 显示成 [1, N]。
// ================================================================
static void print_tensor_info(const OrtApi *api, OrtTypeInfo *type_info)
{
    const OrtTensorTypeAndShapeInfo *tinfo = NULL;
    CHECK_STATUS(api->CastTypeInfoToTensorInfo(type_info, &tinfo));
    if (tinfo == NULL)
    {
        printf("    (non-tensor type)\n");
        return;
    }
    ONNXTensorElementDataType elem_type;
    size_t num_dims = 0;
    CHECK_STATUS(api->GetTensorElementType(tinfo, &elem_type));
    CHECK_STATUS(api->GetDimensionsCount(tinfo, &num_dims));

    printf("    type: %s\n", elem_type_str(elem_type));
    printf("    dims: [");
    if (num_dims > 0)
    {
        // 同时拿数值维度和 symbolic 名字，动态维会打印成 "N" 而不是 -1
        int64_t dims[16];
        const char *sym_dims[16];
        CHECK_STATUS(api->GetDimensions(tinfo, dims, num_dims));
        CHECK_STATUS(api->GetSymbolicDimensions(tinfo, sym_dims, num_dims));
        for (size_t i = 0; i < num_dims; i++)
        {
            if (i)
                printf(", ");
            if (sym_dims[i] && sym_dims[i][0] != '\0') // 有 symbolic 名
                printf("%s", sym_dims[i]);
            else
                printf("%lld", (long long)dims[i]);
        }
    }
    printf("]\n");
}

int main()
{
    // ================================================================
    // 第 0 步: 拿到 OrtApi 入口（和 Lesson 08 完全一样）
    // OrtGetApiBase() 是全局唯一函数，GetApi(27) 返回当前版本的函数表。
    // ================================================================
    const OrtApiBase *api_base = OrtGetApiBase();
    const OrtApi *api = api_base->GetApi(ORT_API_VERSION);

    // ---------- ORT 环境信息 ----------
    printf("========== ORT 环境 ==========\n");
    printf("ORT version: %s\n", api_base->GetVersionString()); // 例: "1.27.0"

    // GetAvailableProviders: 查询这个 ORT 构建里有哪些执行提供者
    // 返回的 providers 是个二维数组 (char**)，用完必须 ReleaseAvailableProviders。
    // 注意: 返回的名字只是"这个构建支持"，不代表一定可用（CUDA 可能没装驱动）。
    char **providers = NULL;
    int provider_count = 0;
    CHECK_STATUS(api->GetAvailableProviders(&providers, &provider_count));
    printf("Available providers: ");
    for (int i = 0; i < provider_count; i++)
        printf("%s%s", i ? ", " : "", providers[i]);
    printf("\n");
    CHECK_STATUS(api->ReleaseAvailableProviders(providers, provider_count));

    // ---------- 创建 Env / Session ----------
    OrtEnv *env = NULL;
    CHECK_STATUS(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "lesson-09", &env));

    OrtSessionOptions *session_options = NULL;
    CHECK_STATUS(api->CreateSessionOptions(&session_options));
    // 第 1 句: 告诉 ORT "我要开到最狠的优化级别"
    CHECK_STATUS(api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL));

    // CreateSession 会把 .onnx 文件解析进内存并做图优化，
    // 内省就是从"优化后的 session"上查信息。
    OrtSession *session = NULL;
    // 第 2 句: 加载模型时, ORT 在这个调用【内部】做图优化
    CHECK_STATUS(api->CreateSession(env, "my_introspect_model.onnx", session_options, &session));
    api->ReleaseSessionOptions(session_options); // 用完立即释放

    // ===============================================================
    //     CreateSession 内部做了这几件事：
    // 1. 解析 .onnx 文件（读 protobuf）
    // 2. 建图（把节点/张量装进内存）
    // 3. 【应用图优化】← 就是这里！
    //    - 常量折叠（把 W、B 这种固定值提前算掉）
    //    - 算子融合（多个算子合成一个）
    //    - 删无用节点
    // 4. 生成可执行计划

    // ================================================================
    // 关键点: 默认 allocator
    // SessionGetInputName 这类函数返回的字符串是 ORT 内部用 allocator
    // 分配的，必须用同一个 allocator 的 Free 释放（不是 free()!）。
    // 这个"谁分配谁释放"规则是整个 C API 内存管理的核心。
    // ================================================================
    OrtAllocator *allocator = NULL;
    CHECK_STATUS(api->GetAllocatorWithDefaultOptions(&allocator));

    // ================================================================
    // 第 1 部分: 图输入
    // 流程 (对每个输入 i):
    //   SessionGetInputCount          -> 先问有几个输入
    //   SessionGetInputName           -> 第 i 个输入叫什么
    //   SessionGetInputTypeInfo       -> 第 i 个输入的 type/shape
    // 拿到 name 和 type_info 用完后分别释放。
    // ================================================================
    printf("\n========== Graph Inputs ==========\n");
    size_t input_count = 0;
    CHECK_STATUS(api->SessionGetInputCount(session, &input_count));
    printf("input count: %zu\n", input_count);
    for (size_t i = 0; i < input_count; i++)
    {
        char *name = NULL;
        OrtTypeInfo *type_info = NULL;
        CHECK_STATUS(api->SessionGetInputName(session, i, allocator, &name));
        CHECK_STATUS(api->SessionGetInputTypeInfo(session, i, &type_info));
        printf("input[%zu] name: %s\n", i, name);
        print_tensor_info(api, type_info);
        allocator->Free(allocator, name); // 名字用 allocator 释放
        api->ReleaseTypeInfo(type_info);  // type_info 用 Release 释放
    }

    // ================================================================
    // 第 2 部分: 可覆盖 initializer (overridable initializer)
    // 概念: 模型里有些权重既声明为 initializer，又声明为 graph input。
    //   这样调用方运行时可以传入新值覆盖默认权重（类似函数的可选参数）。
    // 我们的演示模型里 B 就是这种: 默认值 [0.5,-0.5,1,-1]，
    //   但推理时可以传别的 [4] 数组进来。
    // SessionGetOverridableInitializerCount 只数这类特殊的张量，
    // 普通纯权重（如 W）不在里面。
    // ================================================================
    printf("\n========== Overridable Initializers ==========\n");
    size_t init_count = 0;
    CHECK_STATUS(api->SessionGetOverridableInitializerCount(session, &init_count));
    printf("overridable initializer count: %zu\n", init_count);
    for (size_t i = 0; i < init_count; i++)
    {
        char *name = NULL;
        OrtTypeInfo *type_info = NULL;
        CHECK_STATUS(api->SessionGetOverridableInitializerName(session, i, allocator, &name));
        CHECK_STATUS(api->SessionGetOverridableInitializerTypeInfo(session, i, &type_info));
        printf("initializer[%zu] name: %s\n", i, name);
        print_tensor_info(api, type_info);
        allocator->Free(allocator, name);
        api->ReleaseTypeInfo(type_info);
    }

    // ================================================================
    // 第 3 部分: 图输出 (结构和输入完全对称)
    // ================================================================
    printf("\n========== Graph Outputs ==========\n");
    size_t output_count = 0;
    CHECK_STATUS(api->SessionGetOutputCount(session, &output_count));
    printf("output count: %zu\n", output_count);
    for (size_t i = 0; i < output_count; i++)
    {
        char *name = NULL;
        OrtTypeInfo *type_info = NULL;
        CHECK_STATUS(api->SessionGetOutputName(session, i, allocator, &name));
        CHECK_STATUS(api->SessionGetOutputTypeInfo(session, i, &type_info));
        printf("output[%zu] name: %s\n", i, name);
        print_tensor_info(api, type_info);
        allocator->Free(allocator, name);
        api->ReleaseTypeInfo(type_info);
    }

    // ================================================================
    // 第 4 部分: Model Metadata
    // 相当于模型的"标签页": 谁导出的、什么版本、有没有自定义备注。
    // gen_model.py 里用 helper.set_model_props 塞了 {"author","model_type"}。
    // 部署时可以在模型里藏约定信息（如输入采样率、数据格式），
    // 用 C 代码读出来做运行时判断，不用硬编码。
    // ================================================================
    printf("\n========== Model Metadata ==========\n");
    OrtModelMetadata *metadata = NULL;
    CHECK_STATUS(api->SessionGetModelMetadata(session, &metadata));
    char *value = NULL;
    CHECK_STATUS(api->ModelMetadataGetProducerName(metadata, allocator, &value));
    printf("producer name: %s\n", value);
    allocator->Free(allocator, value);
    CHECK_STATUS(api->ModelMetadataGetGraphName(metadata, allocator, &value));
    printf("graph name:    %s\n", value);
    allocator->Free(allocator, value);
    CHECK_STATUS(api->ModelMetadataGetDomain(metadata, allocator, &value));
    printf("domain:        %s\n", value);
    allocator->Free(allocator, value);
    CHECK_STATUS(api->ModelMetadataGetDescription(metadata, allocator, &value));
    printf("description:   %s\n", value);
    allocator->Free(allocator, value);
    int64_t version = 0;
    CHECK_STATUS(api->ModelMetadataGetVersion(metadata, &version));
    printf("version:       %lld\n", (long long)version);

    // 自定义 metadata 的 key 列表
    // 返回 keys 是一个 char** 数组 + 每个字符串，全部用 allocator 分配，
    // 所以释放时要先逐个 Free 字符串，再 Free 数组本身。
    char **keys = NULL;
    int64_t key_count = 0;
    CHECK_STATUS(api->ModelMetadataGetCustomMetadataMapKeys(metadata, allocator, &keys, &key_count));
    printf("custom keys:   ");
    for (int64_t i = 0; i < key_count; i++)
        printf("%s%s", i ? ", " : "", keys[i]);
    printf(" (%lld)\n", (long long)key_count);
    for (int64_t i = 0; i < key_count; i++)
        allocator->Free(allocator, keys[i]);
    allocator->Free(allocator, keys);
    api->ReleaseModelMetadata(metadata);

    // ================================================================
    // 第 5 步: 清理
    // 逆序释放: 后创建的先释放。
    // 内存释放规则总结（本程序出现的）:
    //   allocator->Free  -> SessionGet*Name / MetadataGet* 返回的字符串和数组
    //   api->Release*    -> type_info / metadata / session / env / status
    // ================================================================
    api->ReleaseSession(session);
    api->ReleaseEnv(env);
    printf("\nDone.\n");
    return 0;
}