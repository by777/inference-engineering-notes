# 02_tvmscript_matmul.py
# Lesson 17: TensorIR / TVMScript - 实验 2: TVMScript 手写 MatMul
#
# 回顾 (实验 1): TVMScript 写 y[i] = x[i] + 1（单层循环）
# 本实验: 手写 [M,K] × [K,N] → [M,N] 矩阵乘法（三层循环）
#
# 数学定义: C[i,j] = Σ_k A[i,k] * B[k,j]
#
# 四个版本对比 —— 同一数学，不同循环写法，性能差异巨大：
#   1. naive:        三层串行循环（i, j, k），最直观 → 基线
#   2. opt:          循环重排（j 放内层）+ T.vectorized 向量化
#   3. parallel:     T.parallel 多核并行（但 k 仍在最内层 → 慢）
#   4. opt_parallel: 并行 × 重排 × 向量化 三刀全上 → 最快
#
# 呼应: Lesson 13 手写调度 / Lesson 14 AutoTVM 自动搜索
# 核心概念: 调度 = 循环变换（顺序/并行/向量化），不改变数学语义，只改性能
#
# 运行: source .venv/bin/activate && python 02_tvmscript_matmul.py
#
# ═══ 前置知识点（遇到不懂先看这里）═══
#
# 【T.serial vs T.vectorized】
#   两者不是"两种循环"，而是同一个循环的两种"注解"——告诉编译器底层生成什么指令：
#   for i in T.serial(N):      # 每次迭代算 1 个元素 → 标量指令
#   for j in T.vectorized(N):  # 每次迭代算 4/8/16 个元素 → SIMD 指令（ARM 上即 NEON）
#   循环结构一样，但生成机器码天差地别。
#
# 【什么循环能 vectorize，什么不能？】
#   循环体                          能向量化吗   原因
#   C[i,j] = A[i,k] * B[k,j]        ✅          每个 j 独立，互不干扰
#   s += A[i,k]（k 循环累加）        ❌          数据依赖（前一步的 s 是下一步的输入）
#   有 if/break 的分支              ❌          控制流不统一，SIMD 难处理
#
# 【为什么 k 放中间、j 放最内层？】(opt 版的核心)
#   固定一组 (i, k)，最内层 j 循环实际上是 rank-1 update（外积累加）:
#     C[i,:] += A[i,k] * B[k,:]
#   - A[i,k] 是标量 → 加载一次进 SIMD 寄存器，广播复用 256 次，零内存访问
#   - B[k,:] / C[i,:] 是整行 → j 方向连续地址，cache line 用满（vs naive 的 1/16）
#   所以 opt 版三个操作数全部最优；naive 版 B 跳跃访问，内存带宽浪费 16 倍。
#
# 【T.parallel 的前提条件】
#   i 循环的不同迭代写 C 的不同行 C[i,:] —— 互不干扰，可以并行
#   （并行 j 也安全；但不能并行 k，因为 k 是累加轴，有数据依赖）

# ========== 环境配置（必须在 import tvm 之前）==========
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tvm-src" / "python"))
sys.path.insert(0, str(_REPO / "tvm-bin"))
os.environ["LD_LIBRARY_PATH"] = (
    str(_REPO / "tvm-bin") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ["TVM_NUM_THREADS"] = "16"  # parallel 循环用到的线程数

import numpy as np
import tvm
from tvm.script import tir as T  # T = TVMScript 方言

M, N, K = 256, 256, 256


# ---------- 1. naive版：三层串行循环（基线） ---------- #
@T.prim_func
def matmul_naive(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    # 第一步：清零 C，因为第二步是 +=（buffer 是裸内存，初始值不确定！）
    for i, j in T.grid(M, N):
        C[i, j] = T.float32(0)

    # 第二步：三层串行循环
    # 循环顺序：i → j → k（k 在最内层）
    # 问题：B[k,j] 在 k 方向跳跃访问（行主序下跨行 = 跳 1KB）
    #   cache line 一次拉 64B，但每次只用 4B → 缓存利用率 ~6%
    #   同样的浪费对每个 (i,j) 都重来一遍 → 内存带宽瓶颈
    for i, j, k in T.grid(M, N, K):
        C[i, j] += A[i, k] * B[k, j]


# ------ 2. opt版：循环重排 + 向量化 ------ #
@T.prim_func
def matmul_opt(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    # 清零循环也可以向量化（逐元素独立操作）
    for i in T.serial(M):
        for j in T.vectorized(N):
            C[i, j] = T.float32(0)

    # 关键优化：把 j 沉到最内层（k 放中间）
    #   - B[k,:] 和 C[i,:] 在 j 方向连续 → cache line 用满
    #   - A[i,k] 是标量 → 广播进 SIMD 寄存器复用，零内存访问
    #   - T.vectorized(N) → 生成 SIMD 指令，一次算 4/8 个 j
    # 这对应 ARM 的 NEON: vfmaq_laneq_f32（标量广播 × 融合乘加）
    for i in T.serial(M):
        for k in T.serial(K):
            for j in T.vectorized(N):
                C[i, j] += A[i, k] * B[k, j]


#  ---------- 3. parallel版：多核并行（但只做了并行）  ---------- #
# ⚠️ 这个版本是个"教学事故"：并行解决了"多核分摊"，
#    但没解决"cache 不友好"——k 还在最内层，B 仍然跳跃访问！
#    结果：16 个核抢带宽跑低效循环，比 opt 还慢 → 证明:
#    "并行不解决 cache 问题，反而放大 cache 问题"
@T.prim_func
def matmul_parallel(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    for i in T.parallel(M):  # 多核并行
        for j in T.vectorized(N):  # 内层 j 向量化 (SIMD)
            C[i, j] = T.float32(0)

    for i in T.parallel(M):  # i 并行: 每行分给一个线程
        for j in T.serial(N):
            for k in T.serial(K):  # ❌ k 在最内层 —— 和 naive 一样跳跃访问 B！
                C[i, j] += A[i, k] * B[k, j]


#  ---------- 4. opt_parallel版：三刀全上（最优）  ---------- #
# 并行(i) × 重排(k中间) × 向量化(j最内) —— 三个优化互不冲突，可以叠加
@T.prim_func
def matmul_opt_parallel(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    for i in T.parallel(M):  # ① i 并行：每行一个线程
        for j in T.vectorized(N):  # ② 清零也向量化
            C[i, j] = T.float32(0)

    for i in T.parallel(M):  # ① 并行
        for k in T.serial(K):  # ② k 中间层（A 广播，B/C 连续）
            for j in T.vectorized(N):  # ③ j 最内层向量化（SIMD）
                C[i, j] += A[i, k] * B[k, j]


#  ------- 5. 编译 4 个版本 ------ #
funcs = {
    "naive": tvm.build(matmul_naive, target="llvm"),
    "opt": tvm.build(matmul_opt, target="llvm"),
    "parallel": tvm.build(matmul_parallel, target="llvm"),
    "opt_parallel": tvm.build(matmul_opt_parallel, target="llvm"),
}
print("编译完成:", {name: str(f) for name, f in funcs.items()})
# 构造随机输入 + numpy 参考结果（golden）
A_np = np.random.rand(M, K).astype("float32")
B_np = np.random.rand(K, N).astype("float32")
golden = A_np @ B_np

# 复用 TVM 张量（避免测时循环里重复转换 numpy → TVM）
A_tvm = tvm.nd.array(A_np)
B_tvm = tvm.nd.array(B_np)


# -------- 6. 运行 + 验证 + 测时 -------- #
def run_and_check(name, f):
    C_tvm = tvm.nd.empty((M, N), dtype="float32")
    f(A_tvm, B_tvm, C_tvm)
    C_np = C_tvm.numpy()

    # 数值验证: 各版本累加顺序不同 → 浮点结果有微小差异，用宽松容差
    # ⚠️ 不要用默认 1e-7，可能误报失败（这是真实工业界的教训）
    np.testing.assert_allclose(C_np, golden, rtol=1e-4, atol=1e-4)

    # warmup 1 次（cache 预热），再计时 10 次取平均
    f(A_tvm, B_tvm, C_tvm)
    t0 = time.perf_counter()
    for _ in range(10):
        f(A_tvm, B_tvm, C_tvm)
    dt = (time.perf_counter() - t0) / 10 * 1000  # ms
    print(f"{name:12s} 耗时 {dt:8.3f} ms  正确性 ✅")
    return dt


results = {name: run_and_check(name, f) for name, f in funcs.items()}
base = results["naive"]
print("\n=== 相对 naive 的加速比 ===")
for name, t in results.items():
    print(f"{name:12s} {t / base:6.2f}x")

# -------- 7. 结论输出：吞吐量 + 关键教训 -------- #
print("\n=== 吞吐量 (GFLOP/s) ===")
# 2 * M * N * K = 乘加各一次，单位: 10^9 次浮点运算 / (时间 ms / 1000)
flops = 2.0 * M * N * K
for name, t in results.items():
    gflops = flops / (t * 1e-3) / 1e9
    print(f"{name:12s} {gflops:8.2f} GFLOP/s")

best_name = min(results, key=results.get)
print(f"\n🏆 最快版本: {best_name} ({results[best_name]:.3f} ms, {base / results[best_name]:.1f}x vs naive)")
print("""
═══ 实验结论（写进笔记）═══

1. 调度 = 循环变换，不改变数学语义，只改性能
   同一段数学，循环怎么写，性能差 37 倍。

2. 三把"刀"：
   循环重排     → cache 友好（B/C 连续、A 广播进寄存器）
   T.vectorized → SIMD 一条指令算一排（ARM 上即 NEON）
   T.parallel   → 多核并行

3. 关键教训：并行不解决 cache 问题，反而放大 cache 问题
   （16 核抢带宽跑低效循环 = 灾难，parallel 比 opt 还慢）

4. 最优组合 = 并行(i) × 重排(k中间) × 向量化(j最内)
   三个优化互不冲突，可以叠加。

5. 实测（256³ float32）:
   naive 6.45ms → opt_parallel 0.17ms，37.7x 加速
""")


# -------- 8. 打印 opt 版 IR, 看循环重排 + 向量化的效果 -------- #
print("=== matmul_opt 的 TIR IR ===")
print(matmul_opt.script())
# 预期看到:
#   for i, k in T.grid(256, 256):
#     for j in T.vectorized(256):
#       C[i, j] = C[i, j] + A[i, k] * B[k, j]
# 注意: T.serial 在 IR 里就是普通 for 循环（serial 是默认属性，不显示）
#       T.vectorized 会显示出来（非默认属性）
