// main.c
// Lesson 12: IoBinding 零拷贝 —— 预分配输入/输出 buffer，避免每次 Run 的分配开销
//
// 对比两种推理方式:
//   普通 Run:    每次调用都要新建 input_tensor / output_tensor (分配+释放)
//   IoBinding:  session 创建后一次性 Bind，之后只 Run，数据直接写进预分配 buffer
//
// 用法:
//   ./main bench <iters>     # 对比普通 Run vs IoBinding 的耗时
//   ./main verify            # 验证两种方式结果一致
//
// 为什么 IoBinding 快?
//   Lesson 11 profile 显示: 一次推理 38us 里 ~11us 是搬运/分配开销 (29%)
//   IoBinding 把"每次分配"变成"一次分配、反复使用"，消除这部分开销。
//   对 CPU 小模型收益可能不大(本来就快)，对 GPU/大模型/流式推理收益显著。
//
// 核心 API:
//   CreateIoBinding(session, &binding)     创建绑定对象
//   BindInput(binding, "X", input_tensor)  绑定输入(名字 + OrtValue)
//   BindOutput(binding, "Y", output_tensor) 绑定输出(名字 + 预分配的 OrtValue)
//   RunWithBinding(session, NULL, binding)  用绑定跑推理
//   GetBoundOutputValues(binding, ...)     取回输出结果
//
// 编译: gcc -std=c17 -D_POSIX_C_SOURCE=199309L -I../ort-bin/include -L../ort-bin/lib -lonnxruntime -Wl,-rpath,../ort-bin/lib -o main main.c

/**
* 方式 A (普通 Run):                          方式 B (IoBinding):
* 每次 Run:                                  绑定一次:
* Run(..., &out_a)   ← ORT 分配 out_a        BindOutput(..., out_b)
* ...用 out_a...                             然后每次:
* ReleaseValue(out_a) ← 释放                 RunWithBinding(...)  ← 直接写 out_buf
*（下次 Run 再分配再释放）                     （无分配无释放, 反复用）

*普通 Run 每次循环的 3 步开销：
*for (int i = 0; i < iters; i++) {
// ① 包一层: CreateTensorWithDataAsOrtValue 创建 OrtValue 对象
//     (这不是内存分配, 是创建一个"结构体对象", 里面有 shape/type 元数据)
// ② 查找:   Run 内部拿名字 "Y" 去图的输出表里查 → 找到输出位置
// ③ 拆包:   GetTensorMutableData / ReleaseValue
*}
*IoBinding 一次性做完这 3 步：
*绑定阶段(一次):                             推理阶段(每次):
*CreateIoBinding(session)                   RunWithBinding(session, NULL, binding)
*BindOutput(binding, "Y", out_b)  ──┐       ← 直接往 out_b 的内存写
*(对象建好了, 名字查好了)           └─→ 记住了, 每次直接用
*缓存的不是内存，是"名字 → 对象"的映射关系。这个映射一旦建立，每次 Run 就不用再查名字、不用再建对象
 *
 */
#define _POSIX_C_SOURCE 199309L // 启用 clock_gettime/CLOCK_MONOTONIC, 必须在所有 include 之前!
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

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void print_output(const char *tag, float *data, int n)
{
    printf("  %s = [", tag);
    for (int i = 0; i < n; i++)
        printf("%s%.3f", i ? ", " : "", data[i]);
    printf("]\n");
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s bench <iters> | verify\n", argv[0]);
        return 1;
    }

    const OrtApiBase *api_base = OrtGetApiBase();
    const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
    printf("ORT version: %s\n", api_base->GetVersionString());

    OrtEnv *env = NULL;
    CHECK_STATUS(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "lesson-12", &env));

    // ---- session (4 线程, 全优化) ----
    OrtSessionOptions *opts = NULL;
    CHECK_STATUS(api->CreateSessionOptions(&opts));
    CHECK_STATUS(api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));
    CHECK_STATUS(api->SetIntraOpNumThreads(opts, 4));
    CHECK_STATUS(api->SetInterOpNumThreads(opts, 4));
    OrtSession *session = NULL;
    CHECK_STATUS(api->CreateSession(env, "mlp3.onnx", opts, &session));
    api->ReleaseSessionOptions(opts);

    // ---- 输入数据 (固定) ----
    static float input_data[256];
    for (int i = 0; i < 256; i++)
        input_data[i] = 0.5f;
    int64_t input_shape[] = {1, 256};
    int64_t output_shape[] = {1, 10};
    const char *input_names[] = {"X"};
    const char *output_names[] = {"Y"};

    // ---- 普通 Run 用的内存信息 ----
    OrtMemoryInfo *mem_info = NULL;
    CHECK_STATUS(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    if (strcmp(argv[1], "verify") == 0)
    {
        // ================================================================
        // 验证: 两种方式结果一致
        // ================================================================
        printf("===== verify: 普通 Run vs IoBinding =====\n");

        // --- 方式 A: 普通 Run ---
        OrtValue *in_a = NULL;
        OrtValue *out_a = NULL;
        CHECK_STATUS(api->CreateTensorWithDataAsOrtValue(mem_info, input_data, sizeof(input_data),
                                                         input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_a));
        CHECK_STATUS(api->Run(session, NULL, input_names, (const OrtValue *const *)&in_a, 1,
                              output_names, 1, &out_a));
        float *out_a_data = NULL;
        CHECK_STATUS(api->GetTensorMutableData(out_a, (void **)&out_a_data));
        print_output("普通 Run   ", out_a_data, 10);

        // --- 方式 B: IoBinding ---
        // 1. 创建 binding 对象（相当于一个"接线板"）
        OrtIoBinding *binding = NULL;
        CHECK_STATUS(api->CreateIoBinding(session, &binding));

        // 2. 把输入插到接线板上: 名字"X" ↔ input_data 包成的 OrtValue
        OrtValue *in_b = NULL;
        CHECK_STATUS(api->CreateTensorWithDataAsOrtValue(mem_info, input_data, sizeof(input_data),
                                                         input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_b));
        CHECK_STATUS(api->BindInput(binding, "X", in_b));

        // 3. 绑定输出: 预分配一块 buffer (10 个 float)
        static float out_buf[10]; // 预分配, 反复使用
        OrtValue *out_b = NULL;
        CHECK_STATUS(api->CreateTensorWithDataAsOrtValue(mem_info, out_buf, sizeof(out_buf),
                                                         output_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &out_b));
        CHECK_STATUS(api->BindOutput(binding, "Y", out_b));

        // 4. RunWithBinding 注意: 不传名字数组, 只传 binding!
        CHECK_STATUS(api->RunWithBinding(session, NULL, binding));
        print_output("IoBinding  ", out_buf, 10); // 数据直接写进 out_buf

        // 5. 清理 binding (注意: binding 释放不释放绑定的 OrtValue, 要单独 Release)
        api->ReleaseIoBinding(binding);
        api->ReleaseValue(in_b);
        api->ReleaseValue(out_b);
        api->ReleaseValue(in_a);
        api->ReleaseValue(out_a);
        printf("verify done.\n");
    }
    else if (strcmp(argv[1], "bench") == 0)
    {
        int iters = atoi(argv[2]);

        // ---- 方式 A: 普通 Run (每次分配) ----
        printf("===== bench %d iters =====\n", iters);
        OrtValue *in_a = NULL;
        OrtValue *out_a = NULL;
        CHECK_STATUS(api->CreateTensorWithDataAsOrtValue(mem_info, input_data, sizeof(input_data),
                                                         input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_a));
        api->Run(session, NULL, input_names, (const OrtValue *const *)&in_a, 1, output_names, 1, &out_a);
        api->ReleaseValue(out_a); // 预热

        double t0 = now_ms();
        for (int i = 0; i < iters; i++)
        {
            OrtValue *out = NULL;
            CHECK_STATUS(api->Run(session, NULL, input_names, (const OrtValue *const *)&in_a, 1,
                                  output_names, 1, &out)); // 每次新建 out
            api->ReleaseValue(out);                        // 每次释放 out
        }
        double t_run = now_ms() - t0;
        api->ReleaseValue(in_a);
        printf("普通 Run:  avg = %.3f ms/iter\n", t_run / iters);

        // ---- 方式 B: IoBinding (一次性绑定, 反复跑) ----
        OrtIoBinding *binding = NULL;
        CHECK_STATUS(api->CreateIoBinding(session, &binding));
        OrtValue *in_b = NULL;
        CHECK_STATUS(api->CreateTensorWithDataAsOrtValue(mem_info, input_data, sizeof(input_data),
                                                         input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_b));
        CHECK_STATUS(api->BindInput(binding, "X", in_b));
        static float out_buf[10];
        OrtValue *out_b = NULL;
        CHECK_STATUS(api->CreateTensorWithDataAsOrtValue(mem_info, out_buf, sizeof(out_buf),
                                                         output_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &out_b));
        CHECK_STATUS(api->BindOutput(binding, "Y", out_b));

        CHECK_STATUS(api->RunWithBinding(session, NULL, binding)); // 预热

        t0 = now_ms();
        for (int i = 0; i < iters; i++)
        {
            CHECK_STATUS(api->RunWithBinding(session, NULL, binding)); // 无分配, 直接算
        }
        double t_bind = now_ms() - t0;

        api->ReleaseIoBinding(binding);
        api->ReleaseValue(in_b);
        api->ReleaseValue(out_b);

        printf("IoBinding: avg = %.3f ms/iter\n", t_bind / iters);
        printf("提升: %.1f%%\n", (t_run - t_bind) / t_run * 100.0);
    }
    else
    {
        printf("Unknown mode: %s\n", argv[1]);
        return 1;
    }

    api->ReleaseMemoryInfo(mem_info);
    api->ReleaseSession(session);
    api->ReleaseEnv(env);
    printf("Done.\n");
    return 0;
}