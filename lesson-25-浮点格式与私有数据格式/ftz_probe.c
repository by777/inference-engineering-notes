/*
 * ftz_probe.c —— 探测"subnormal 会不会被硬件/编译器刷成 0"
 *
 * 背景（对应 lesson-25 第 02 章）：
 *   IEEE754 默认要求支持 subnormal（渐进下溢），但硬件有 FTZ/DAZ 开关、
 *   编译器有 -ffast-math 之类的选项，都可能在你不注意时把 subnormal 干掉。
 *
 * 两个概念要分清（本程序会分别演示）：
 *   DAZ (Denormals Are Zero)  —— 输入侧：subnormal 被 FPU **读入**时当 0
 *   FTZ (Flush To Zero)       —— 结果侧：运算**结果**是 subnormal 时刷成 0
 *
 * 用法：
 *   gcc -O2 ftz_probe.c -o probe && ./probe
 *   gcc -O2 -ffast-math ftz_probe.c -o probe_fast && ./probe_fast
 *
 * 交叉编译到 ARM：
 *   aarch64-linux-gnu-gcc -O2 ftz_probe.c -o probe_arm64
 *   （AArch64 的 FZ 位在 FPCR bit 24，本程序会打印）
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define HAVE_MXCSR 1
#else
#define HAVE_MXCSR 0
#endif

/* 读内存里的真实位模式（memcpy 不经过 FPU，所以不受 DAZ 影响） */
static unsigned raw_bits(float x)
{
    unsigned u;
    memcpy(&u, &x, sizeof u);
    return u;
}

/* 按位模式判断类别：注意这里只读 bits，不做浮点运算 */
static const char *classify(unsigned u)
{
    unsigned e = (u >> 23) & 0xFFu;   /* float32 的 8 位指数字段 */
    unsigned m = u & 0x7FFFFFu;
    if (e == 0)          return m ? "SUBNORMAL" : "±0";
    if (e == 0xFFu)      return m ? "NaN" : "±inf";
    return "正规数";
}

/* 打印：位模式（内存真相）+ 数值（经 FPU，可能被 DAZ 影响） */
static void dump(const char *label, float x)
{
    unsigned u = raw_bits(x);
    printf("  %-20s bits=0x%08X [%-9s]  printf=%-.6e\n",
           label, u, classify(u), (double)x);
}

#if defined(__aarch64__)
static unsigned long read_fpcr(void)
{
    unsigned long v;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(v));
    return v;
}
#endif

/* 运行时强制恢复 IEEE754 合规模式（清掉 FTZ/DAZ 或 FZ）。
 * 用途：如果你必须用 -ffast-math 换性能，可以在启动时调一次这个，
 *       把 subnormal 支持要回来。 */
static void force_ieee(void)
{
#if HAVE_MXCSR
    unsigned csr = (unsigned)_mm_getcsr();
    csr &= ~((1u << 15) | (1u << 6));          /* 清 FTZ(15) 和 DAZ(6) */
    _mm_setcsr(csr);
#elif defined(__aarch64__)
    unsigned long fpcr = read_fpcr();
    fpcr &= ~((1UL << 24) | (1UL << 19));      /* 清 FZ(24) 和 FZ16(19) */
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#endif
}

int main(int argc, char **argv)
{
    int want_ieee = (argc > 1 && strcmp(argv[1], "--ieee") == 0);
    if (want_ieee) {
        force_ieee();
        printf(">>> --ieee：运行时已清掉 FTZ/DAZ，强制 IEEE754 模式\n\n");
    }

    printf("=== 0. 环境 ===\n");
#if defined(__x86_64__)
    printf("  架构: x86_64\n");
#elif defined(__i386__)
    printf("  架构: i386\n");
#elif defined(__aarch64__)
    printf("  架构: aarch64\n");
#elif defined(__arm__)
    printf("  架构: arm (32-bit)\n");
#else
    printf("  架构: 未知\n");
#endif
    printf("  编译器: %s %d.%d\n",
#if defined(__clang__)
           "clang", __clang_major__, __clang_minor__
#elif defined(__GNUC__)
           "gcc", __GNUC__, __GNUC_MINOR__
#else
           "unknown", 0, 0
#endif
    );
#ifdef __FAST_MATH__
    printf("  __FAST_MATH__            = 已开启 ⚠️\n");
#else
    printf("  __FAST_MATH__            = 未开启\n");
#endif
#ifdef __FINITE_MATH_ONLY__
    printf("  __FINITE_MATH_ONLY__     = %d\n", __FINITE_MATH_ONLY__);
#endif

    /* ---- 1. 位模式（内存真相）：memcpy 不经 FPU ---- */
    printf("\n=== 1. 内存里的位模式（memcpy，不经 FPU） ===\n");
    {
        volatile float a = 1e-40f, b = 1e-45f;
        float x = a, y = b;
        dump("1e-40f", x);
        dump("1e-45f (最小subnormal)", y);
    }
    printf("  ^ 这一节看 bits 列：位模式在不在。\n");

    /* ---- 2. DAZ 演示：subnormal 过一次 FPU ---- */
    printf("\n=== 2. DAZ 演示：subnormal 被 FPU 读入 ===\n");
    {
        volatile float s = 1e-40f;
        float in = s;
        /* 用 in + in 而不是 in + 0.0f：后者会被 -ffast-math（含 -fno-signed-zeros）
         * 优化成恒等操作，测不到东西。in + in = 2e-40 仍是 subnormal。 */
        float twice = in + in;
        printf("  运算前   bits = 0x%08X [%s]\n", raw_bits(in), classify(raw_bits(in)));
        printf("  in + in  bits = 0x%08X [%s]\n", raw_bits(twice), classify(raw_bits(twice)));
        printf("  -> %s\n", raw_bits(twice) == 0
               ? "被刷成 0：DAZ 生效（输入侧 subnormal 被当 0）"
               : "仍是 subnormal：DAZ 关闭，IEEE 合规（期望 2e-40）");
    }

    /* ---- 3. FTZ 演示：运算结果是 subnormal ---- */
    printf("\n=== 3. FTZ 演示：结果侧下溢（量化 s_w * s_x 就是这个场景） ===\n");
    {
        volatile float a1 = 1e-20f, a2 = 1e-20f;
        volatile float b1 = 1e-25f, b2 = 1e-25f;
        float p = a1 * a2;          /* 1e-40 -> subnormal */
        float q = b1 * b2;          /* 1e-50 -> 低于最小 subnormal */
        dump("1e-20 * 1e-20", p);
        dump("1e-25 * 1e-25", q);
        printf("  -> 前者是 %s\n",
               raw_bits(p) == 0 ? "0（FTZ 生效，本应是 subnormal）"
                                : "subnormal（FTZ 关闭）");
    }

    /* ---- 4. 硬件状态位 ---- */
#if HAVE_MXCSR
    printf("\n=== 4. x86 MXCSR 状态位 ===\n");
    {
        unsigned csr = (unsigned)_mm_getcsr();
        printf("  MXCSR = 0x%08X\n", csr);
        printf("  bit15 FTZ = %u  %s\n", (csr >> 15) & 1u,
               ((csr >> 15) & 1u) ? "⚠️ 结果侧刷零" : "关（正常）");
        printf("  bit 6 DAZ = %u  %s\n", (csr >> 6) & 1u,
               ((csr >> 6) & 1u) ? "⚠️ 输入侧当零" : "关（正常）");
    }
#elif defined(__aarch64__)
    printf("\n=== 4. AArch64 FPCR 状态位 ===\n");
    {
        unsigned long fpcr = read_fpcr();
        printf("  FPCR = 0x%016lX\n", fpcr);
        printf("  bit24 FZ = %lu  %s\n", (fpcr >> 24) & 1UL,
               ((fpcr >> 24) & 1UL) ? "⚠️ Flush-to-zero 生效" : "关（正常）");
        printf("  bit19 FZ16 = %lu （仅影响 fp16 运算）\n", (fpcr >> 19) & 1UL);
    }
#else
    printf("\n=== 4. 本架构未实现状态位读取 ===\n");
#endif

    return 0;
}
