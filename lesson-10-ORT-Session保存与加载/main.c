// main.c
// Lesson 10: Session 保存/加载 —— 把优化后的模型存下来，下次直接加载
// 用法:
//   ./main save <src.onnx> <dst.onnx>   # 加载原始模型并保存"优化后"的图
//   ./main run  <model.onnx>            # 加载模型跑一次推理（验证结果）
//   ./main ort  <src.onnx> <dst.ort>    # 另存为 ORT 专属格式（体积小、加载快）
//
// 为什么有用?
//   每次 CreateSession 都要重新做图优化（耗时）。边缘设备上把优化好的
//   模型保存一次，部署时直接加载，省掉重复优化时间。
//   保存格式有两种:
//     - .onnx: 普通 ONNX 格式，但图已被 ORT 优化（比如算子融合）
//     - .ort : ORT 专属格式，只含 ORT 需要的运行时信息，加载更快
//
// 编译: gcc -std=c17 -I../ort-bin/include -L../ort-bin/lib -lonnxruntime -Wl,-rpath,../ort-bin/lib -o main main.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnxruntime_c_api.h"

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

static void run_inference(const OrtApi *api, OrtSession *session)
{
    // ---- 构造输入 ----
    float input_data[8] = {1., 1., 1., 1.,
                           1., 1., 1., 1.};
    int64_t input_shape[2] = {1, 8};
    OrtValue *input_tensor = NULL;
    OrtMemoryInfo *mem_info = NULL;
    CHECK_STATUS(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));
    CHECK_STATUS(api->CreateTensorWithDataAsOrtValue(mem_info, input_data, sizeof(input_data),
                                                     input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                     &input_tensor));
    api->ReleaseMemoryInfo(mem_info);

    // ====== run inference ======
    const char *input_names[] = {"X"};
    const char *output_names[] = {"Y"};
    OrtValue *output_tensor = NULL;
    // onnxruntime会自己为输出分配内存，返回 OrtValue*，里面包含 shape + 数据指针
    CHECK_STATUS(api->Run(session, NULL,
                          input_names, (const OrtValue *const *)&input_tensor, 1,
                          output_names, 1, &output_tensor));
    float *out = NULL;
    CHECK_STATUS(
        api->GetTensorMutableData(output_tensor, (void **)&out));
    printf("  output Y = [");
    for (int i = 0; i < 4; i++)
        printf("%s%.1f", i ? ", " : "", out[i]);
    printf("]\n");

    api->ReleaseValue(output_tensor);
    api->ReleaseValue(input_tensor);
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        printf("Usage: %s save <src.onnx> <dst.onnx>\n", argv[0]);
        printf("       %s run  <model.onnx>\n", argv[0]);
        printf("       %s ort  <src.onnx> <dst.ort>\n", argv[0]);
        return 1;
    }
    const char *mode = argv[1];
    const OrtApiBase *api_base = OrtGetApiBase();
    const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
    printf("ORT version: %s\n", api_base->GetVersionString());

    OrtEnv *env = NULL;
    CHECK_STATUS(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

    // ================================================================
    // 模式 1: save —— 加载原始模型，把优化后的图存成 .onnx
    // 核心 API: SetOptimizedModelFilePath(options, 路径)
    //   CreateSession 内部做完图优化后，会把优化后的图写到这个路径。
    // ================================================================
    if (strcmp(mode, "save") == 0)
    {
        if (argc < 4)
        {
            printf("need <src.onnx> <dst.onnx>\n");
            return 1;
        }
        const char *src = argv[2];
        const char *dst = argv[3];

        OrtSessionOptions *opts = NULL;
        CHECK_STATUS(api->CreateSessionOptions(&opts));
        CHECK_STATUS(api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));
        // 关键一行: 告诉 ORT "优化完把图存到 dst"
        CHECK_STATUS(api->SetOptimizedModelFilePath(opts, dst));

        OrtSession *session = NULL;
        CHECK_STATUS(api->CreateSession(env, src, opts, &session));
        api->ReleaseSessionOptions(opts);

        printf("Optimized model saved to: %s\n", dst);
        // 顺手跑一次推理，验证保存出来的模型语义没变
        run_inference(api, session);
        api->ReleaseSession(session);
    }
    // ================================================================
    // 模式 2: ort —— 另存为 ORT 专属格式
    // 通过 session config 指定保存格式:
    //   key   = "session.save_model_format"
    //   value = "ort"
    // .ort 格式: 只含运行时信息，体积更小、加载更快，但不能被 onnx
    // 工具链读取，也不能跨 ORT 大版本（必须在同版本加载）。
    // ================================================================
    else if (strcmp(mode, "ort") == 0)
    {
        if (argc < 4)
        {
            printf("need <src.onnx> <dst.ort>\n");
            return 1;
        }
        const char *src = argv[2];
        const char *dst = argv[3];

        OrtSessionOptions *opts = NULL;
        CHECK_STATUS(api->CreateSessionOptions(&opts));
        // 注意: 保存 .ort 格式不能用 ORT_ENABLE_ALL(99)!
        // ORT_ENABLE_ALL 会启用 Level3+ 布局优化(NchwcTransformer),
        // 这类硬件特定优化序列化到 .ort 后加载会报 "ORT model verification failed".
        // .ort 格式的设计上限就是 ORT_ENABLE_EXTENDED(2).
        CHECK_STATUS(api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_EXTENDED));
        CHECK_STATUS(api->SetOptimizedModelFilePath(opts, dst));
        // 关键: 指定保存为 ORT 格式（默认是 onnx）
        // 注意: 值必须是 "ORT" 大写! 头文件注释: 'ORT' (case sensitive)
        // 传小写 "ort" 不会被识别 → ORT 会退回默认的 ONNX 格式保存,
        // 导致文件内容是 protobuf 但扩展名 .ort, 加载时按 flatbuffer 解析
        // 会报 "ORT model verification failed".
        CHECK_STATUS(api->AddSessionConfigEntry(opts, "session.save_model_format", "ORT"));
        // 一些其他配置键（来自 session_options_config_keys.h）
        // AddSessionConfigEntry(opts, "session.intra_op.allow_spinning", "0");   // 禁止线程池自旋
        // AddSessionConfigEntry(opts, "session.use_env_allocators", "1");        // 复用环境分配器
        // AddSessionConfigEntry(opts, "session.graph_optimization_level", "2");  // 用字符串设优化级别

        OrtSession *session = NULL;
        CHECK_STATUS(api->CreateSession(env, src, opts, &session));
        api->ReleaseSessionOptions(opts);

        printf("ORT-format model saved to: %s\n", dst);
        run_inference(api, session);
        api->ReleaseSession(session);
    }
    // ================================================================
    // 模式 3: run —— 直接加载（可能是优化后的模型），跑推理
    // 注意: 加载优化后的模型时不再需要 SetOptimizedModelFilePath，
    // 普通 CreateSession 就行。这也验证了保存的模型能正常加载。
    // ================================================================
    else if (strcmp(mode, "run") == 0)
    {
        if (argc < 3)
        {
            printf("need <model.onnx>\n");
            return 1;
        }
        const char *model_path = argv[2];

        OrtSessionOptions *opts = NULL;
        CHECK_STATUS(api->CreateSessionOptions(&opts));
        CHECK_STATUS(api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));

        OrtSession *session = NULL;
        CHECK_STATUS(api->CreateSession(env, model_path, opts, &session));
        api->ReleaseSessionOptions(opts);

        printf("Loaded model: %s\n", model_path);
        run_inference(api, session);
        api->ReleaseSession(session);
    }
    else
    {
        printf("Unknown mode: %s\n", mode);
        return 1;
    }

    api->ReleaseEnv(env);
    printf("Done.\n");
    return 0;
}