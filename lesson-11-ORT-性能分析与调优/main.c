// main.c
// Lesson 11: 性能分析与调优
// 用法:
//   ./main bench <iters> <intra_threads> [inter_threads]
//       用指定线程数跑 <iters> 次推理，打印平均耗时（毫秒）
//   ./main profile <iters>
//       开启 profiling 跑 <iters> 次，导出 json 文件（找热点算子用）
//
// 核心知识点:
//   1. SetIntraOpNumThreads   每个算子内部并行线程数（MatMul 多线程靠它）
//   2. SetInterOpNumThreads   并行执行多个算子的线程数（图并行，分支多才有效）
//   3. EnableProfiling        SessionOptions 里开启，跑完后 SessionEndProfiling 拿 json
//   4. 线程不是越多越好: 小模型线程开销 > 收益，要实测
// 这节课教你用数据说话：bench 测出最优线程数，profile 找出热点算子，而不是凭经验瞎调。
// 编译: gcc -std=c17 -I../ort-bin/include -L../ort-bin/lib -lonnxruntime -Wl,-rpath,../ort-bin/lib -o main main.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
// 毫秒计时器（CLOCK_MONOTONIC 单调时钟，不受系统时间调整影响）
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// 创建Session返回给调用方 - 线程配置的关键
static OrtSession *create_session(const OrtApi *api, OrtEnv *env,
                                  int intra_threads, int inter_threads, int enable_profiling)
{
    OrtSessionOptions *opts = NULL;
    CHECK_STATUS(api->CreateSessionOptions(&opts));
    CHECK_STATUS(api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));
    //  算子内线程: MatMul/Gemm 内部并行
    CHECK_STATUS(api->SetIntraOpNumThreads(opts, intra_threads));
    // 算子间线程: 图级并行（本模型是串行链，作用不大，但演示设置）
    CHECK_STATUS(api->SetInterOpNumThreads(opts, inter_threads));
    if (enable_profiling)
    {
        CHECK_STATUS(api->EnableProfiling(opts, "profile.json"));
    }

    OrtSession *session = NULL;
    CHECK_STATUS(api->CreateSession(env, "mlp3.onnx", opts, &session));
    api->ReleaseSessionOptions(opts);
    return session;
}

static void run_once(const OrtApi *api, OrtSession *session)
{
    static float input_data[256];
    static int initialized = 0;
    if (!initialized)
    {
        for (int i = 0; i < 256; i++)
            input_data[i] = 0.5f;
        initialized = 1;
    }
    int64_t input_shape[2] = {1, 256};
    OrtMemoryInfo *mem_info = NULL;
    OrtValue *input_tensor = NULL;
    OrtValue *output_tensor = NULL;
    CHECK_STATUS(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));
    CHECK_STATUS(api->CreateTensorWithDataAsOrtValue(
        mem_info, input_data, sizeof(input_data),
        input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
    api->ReleaseMemoryInfo(mem_info);
    const char *input_names[] = {"X"};
    const char *output_names[] = {"Y"};
    CHECK_STATUS(api->Run(session, NULL,
                          input_names, (const OrtValue *const *)&input_tensor, 1,
                          output_names, 1, &output_tensor));

    api->ReleaseValue(output_tensor);
    api->ReleaseValue(input_tensor);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage:\n");
        printf("  %s bench <iters> <intra_threads> [inter_threads]\n", argv[0]);
        printf("  %s profile <iters>\n", argv[0]);
        return 1;
    }
    const char *mode = argv[1];

    const OrtApiBase *api_base = OrtGetApiBase();
    const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
    printf("ORT version: %s\n", api_base->GetVersionString());

    OrtEnv *env = NULL;
    CHECK_STATUS(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "lesson-11", &env));
    if (strcmp(mode, "bench") == 0)
    {
        int iters = atoi(argv[2]);
        int intra = atoi(argv[3]);
        int inter = (argc >= 5) ? atoi(argv[4]) : intra;
        printf("Config: iters=%d, intra_threads=%d, inter_threads=%d\n", iters, intra, inter);

        OrtSession *session = create_session(api, env, intra, inter, 0);

        // 预热: 第一次 Run 有内存池初始化、kernel 加载等一次性开销，不计入统计
        run_once(api, session);

        double t0 = now_ms();
        for (int i = 0; i < iters; i++)
            run_once(api, session);
        double elapsed = now_ms() - t0;

        printf("Total:   %.2f ms (%d iters)\n", elapsed, iters);
        printf("Average: %.3f ms / iter\n", elapsed / iters);
        printf("Throughput: %.1f iters / sec\n", iters / (elapsed / 1000.0));

        api->ReleaseSession(session);
    }
    else if (strcmp(mode, "profile") == 0)
    {
        int iters = atoi(argv[2]);
        printf("Profiling %d iters ...\n", iters);

        OrtSession *session = create_session(api, env, 4, 4, 1);
        run_once(api, session); // 预热

        for (int i = 0; i < iters; i++)
            run_once(api, session);

        // 结束 profiling，返回生成的 json 文件名（用 allocator 分配，要释放）
        OrtAllocator *allocator = NULL;
        CHECK_STATUS(api->GetAllocatorWithDefaultOptions(&allocator));
        char *profile_file = NULL;
        // 停止并导出 json 文件
        CHECK_STATUS(api->SessionEndProfiling(session, allocator, &profile_file));
        printf("Profile saved to: %s\n", profile_file);
        allocator->Free(allocator, profile_file);

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