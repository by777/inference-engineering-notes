# 01_manual_schedule.py
# Lesson 14: AutoTVM 自动调优 - 实验 1: 手动调度的 GEMM 基准
#
# 目标：用手写 schedule（tile+vectorize+parallel）得到一个"人工最优"
#       供后续 AutoTVM 对比
#
# 运行: python 01_manual_schedule.py

import time
import numpy as np
import tvm
from tvm import te

# ----------- 1.定义GEMM计算 ----------- #
M, N, K = 512, 512, 512
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")


# ----------- 2.手动调度 ----------- #
s = te.create_schedule(C.op)

# 分块： i->(io, ii), j->(jo, ji) 块大小16
factor = 16  # factor=16 的意思是每块 16 个元素
# io = 外层循环变量：for io in range(32)，控制"第几块"
# ii = 内层循环变量：for ii in range(16)，控制"块内第几个"
io, ii = s[C].split(C.op.axis[0], factor=factor)
jo, ji = s[C].split(C.op.axis[1], factor=factor)
# 重排: 外层先扫块，内层再算块内
# 如果顺序写错会怎样？
# 正确性不受影响（结果一样，误差不变），但性能可能变差

# 这种写法1，考虑到了行优先
# for (ii)                // 慢变
#   for (ji)              // 快变  ← j 变化最快
#     C[i*16+ii][j*16+ji] = ...
s[C].reorder(io, jo, ii, ji)

# 写法2，C 写入跨行、A 读取跨行，确实差
# for (ji)                // 慢变
#   for (ii)              // 快变  ← i 变化最快
#     C[i*16+ii][j*16+ji] = ...
# s[C].reorder(jo, io, ji, ii)


# 向量化内层
s[C].vectorize(ji)
# 并行化外层
s[C].parallel(io)

# C[M][N] 行优先:
#   axis[0] = i = 行轴（慢变维）→ 块内 ii 不连续 ❌
#   axis[1] = j = 列轴（快变维）→ 块内 ji 连续 ✅

# split(axis[0]) → ii 块内: 地址跳 N    → 不连续
# split(axis[1]) → ji 块内: 地址步长 1  → 连续 ✓

# reorder(io, jo, ii, ji) 让 ji 在最内层   ← 写入连续
# vectorize(ji)                            ← SIMD 连续 load

# ---------- 3. 编译 ---------- #
print("编译中...")
f = tvm.build(s, [A, B, C], target="llvm")
print("编译完成")


# ---------- 4. 基准测试 ---------- #
a_np = np.random.rand(M, K).astype("float32")
b_np = np.random.rand(K, N).astype("float32")
a_tvm, b_tvm = tvm.nd.array(a_np), tvm.nd.array(b_np)
c_tvm = tvm.nd.empty((M, N), dtype="float32")

# 预热
f(a_tvm, b_tvm, c_tvm)

# 计时 (100 次)
t0 = time.time()
for _ in range(100):
    f(a_tvm, b_tvm, c_tvm)
t1 = time.time()
avg_ms = (t1 - t0) / 100 * 1000
print(f"手动调度 GEMM(512x512x512): {avg_ms:.3f} ms")

# 验证正确性
c_np = np.matmul(a_np, b_np)
print("误差:", np.max(np.abs(c_tvm.numpy() - c_np)))
