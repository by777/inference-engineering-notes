// main.c
// Lesson 08: 纯 C API 推理 — 不依赖 C++ 包装层
// 编译: gcc -std=c17 -I../ort-bin/include -L../ort-bin/lib -lonnxruntime -Wl,-rpath,../ort-bin/lib -o main main.c

// 1. 获取 OrtApi 入口  ← 唯一全局函数
// 2. 创建 Environment  ← 日志+全局状态
// 3. 创建 SessionOptions  ← 配置
// 4. 创建 Session  ← 加载模型
// 5. 准备输入数据  ← 构造输入 Tensor
// 6. 准备输出变量
// 7. Run  ← 执行推理
// 8. 获取输出数据 ← 读出结果 shape + 数值
// 9. 打印 + 验证
// 10. 清理  ← 逆序释放

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnxruntime_c_api.h"

// 辅助宏: 检查 OrtStatus 并退出
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

int main()
{
    // ========== 1. 获取 OrtApi 入口 ==========
    const OrtApiBase *api_base = OrtGetApiBase();
    // 返回 OrtApiBase 结构体（只有两个函数指针：GetApi 和 GetVersionString）
    const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
    printf("ORT version: %s\n", api_base->GetVersionString());

    // ========== 2. 创建 Environment ==========
    // OrtEnv 是 ORT 的全局上下文，通常进程只创建一个。
    //     ORT_LOGGING_LEVEL_WARNING — 只打印 WARNING 及以上级别的日志
    // "lesson-08" — 日志前缀，方便在大日志里过滤
    OrtEnv *env = NULL;
    CHECK_STATUS(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "lesson-08", &env));

    // ========== 3. 创建 SessionOptions 并配置 ==========
    OrtSessionOptions *session_options = NULL;
    // 创建配置对象
    CHECK_STATUS(api->CreateSessionOptions(&session_options));
    // 图优化级别
    CHECK_STATUS(api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL));
    // 算子内线程数
    CHECK_STATUS(api->SetIntraOpNumThreads(session_options, 1));
    // 算子间线程数
    CHECK_STATUS(api->SetInterOpNumThreads(session_options, 1));

    // ========== 4. 创建 Session ==========
    OrtSession *session = NULL;
    CHECK_STATUS(api->CreateSession(env, "my_test_model.onnx", session_options, &session));
    // 创建 OrtSession 即加载模型（解析 .onnx 文件、做图优化）。
    // 参数传进去之后 session_options 就没用了，可以立刻释放。
    api->ReleaseSessionOptions(session_options);

    // ========== 5. 准备输入数据 ==========
    // 输入 shape: [1, 4], float32, 值: [1.0, 2.0, 3.0, 4.0]
    int64_t input_shape[] = {1, 4};
    size_t input_count = 4;
    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t data_bytes = input_count * sizeof(float);

    // 创建 CPU MemoryInfo
    OrtMemoryInfo *memory_info = NULL;
    CHECK_STATUS(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    // 创建输入 OrtValue (p_data 由调用者管理, ReleaseValue 不会释放它)
    OrtValue *input_tensor = NULL;
    CHECK_STATUS(api->CreateTensorWithDataAsOrtValue(
        memory_info,                         // 内存在哪（CPU / GPU）
        input_data,                          // 数据指针
        data_bytes,                          // 数据字节数
        input_shape,                         // shape 数组
        2,                                   // shape 维数(shape[] 长度)
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, // 数据类型
        &input_tensor                        // 返回的 OrtValue*
        ));
    // MemoryInfo 在创建 Tensor 后可以释放
    api->ReleaseMemoryInfo(memory_info);

    // ========== 6. 准备输出 ==========
    const char *input_names[] = {"X"}; // 跟 onnx 图里的输入名一致
    const char *output_names[] = {"Y"};
    OrtValue *output_tensor = NULL;

    // ========== 7. 运行推理 ==========
    CHECK_STATUS(api->Run(session,
                          NULL,
                          input_names,
                          (const OrtValue *const *)&input_tensor,
                          1,
                          output_names,
                          1,
                          &output_tensor));

    // ========== 8. 获取输出数据 ==========
    // 获取 tensor 类型和 shape 信息
    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    CHECK_STATUS(api->GetTensorTypeAndShape(output_tensor, &shape_info));

    ONNXTensorElementDataType elem_type;
    CHECK_STATUS(api->GetTensorElementType(shape_info, &elem_type));

    size_t num_dims;
    CHECK_STATUS(api->GetDimensionsCount(shape_info, &num_dims));

    int64_t out_shape[4];
    CHECK_STATUS(api->GetDimensions(shape_info, out_shape, 4));

    size_t total_elements;
    CHECK_STATUS(api->GetTensorShapeElementCount(shape_info, &total_elements));
    api->ReleaseTensorTypeAndShapeInfo(shape_info);

    // 获取数据指针
    float *output_data = NULL;
    CHECK_STATUS(api->GetTensorMutableData(output_tensor, (void **)&output_data));

    // ========== 9. 打印结果 ==========
    printf("Output shape: [");
    for (size_t i = 0; i < num_dims; i++)
        printf("%s%lld", i ? ", " : "", (long long)out_shape[i]);
    printf("]\n");
    printf("Element type: %d (1=float)\n", elem_type);
    printf("Total elements: %zu\n", total_elements);
    printf("Output values: [");
    for (size_t i = 0; i < total_elements; i++)
        printf("%s%.2f", i ? ", " : "", output_data[i]);
    printf("]\n");

    // 验证: 预期 Y = X * 2 + 1 → [3.0, 5.0, 7.0, 9.0]
    float expected[] = {3.0f, 5.0f, 7.0f, 9.0f};
    int pass = 1;
    for (size_t i = 0; i < total_elements; i++)
    {
        if (output_data[i] != expected[i])
        {
            printf("Mismatch at [%zu]: got %.2f, expected %.2f\n",
                   i, output_data[i], expected[i]);
            pass = 0;
        }
    }
    printf("%s\n", pass ? "✅ PASS" : "❌ FAIL");

    // ========== 10. 清理 ==========
    api->ReleaseValue(output_tensor);
    api->ReleaseValue(input_tensor);
    api->ReleaseSession(session);
    api->ReleaseEnv(env);

    return pass ? 0 : 1;
}