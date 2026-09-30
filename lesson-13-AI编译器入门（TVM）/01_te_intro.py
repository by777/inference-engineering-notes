# 01_te_intro.py
# Lesson 13: AI 编译器入门（TVM） - 实验1：TE手写算子

# 核心思想：计算与调度分离（Halide思想。TVM的根基）
# 计算（Compute）：描述“算什么” ——数学定义
# 调度（Schedule）：描述“怎么算” - 性能优化（分块/向量化/并行）
# 同一个compute，可以配不同的Schedule，生成不同性能的代码

# 运行： python 01_te_intro.py

# 什么是te: Tensor Expression, 张量表达式
"""
tvm (整个包，AI 编译器框架)
│
├── tvm.te        ← Tensor Expression: 手写算子用的 DSL（你现在学的）
├── tvm.relay     ← 图 IR: 从 ONNX 等导入的计算图（实验 2 会用到）
├── tvm.tir       ← 底层 IR: 循环、内存、指令级别的中间表示
├── tvm.target    ← 目标平台描述 (llvm/cuda/arm...)
├── tvm.build     ← 编译入口（把 TE/Relay 变成机器码）
└── tvm.nd / tvm.runtime ← 张量容器和执行运行时
"""

import tvm
from tvm import te
import numpy as np

# ============== 1. 最简单的算子：y[i] = x[i] + 1 ==============
print("===== 实验 1.1: 最简单的算子 y[i] = x[i] + 1 =====")

# 计算：描述“算什么”
n = 8
x = te.placeholder((n,), name="x")  # 输入张量
y = te.compute((n,), lambda i: x[i] + 1, name="y")  # 输出张量

# 调度：描述“怎么算”
s = te.create_schedule(y.op)


# 编译：生成机器码
f = tvm.build(s, [x, y], target="llvm")


# 执行
x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype="float32")
x_tvm = tvm.nd.array(x_np)
y_tvm = tvm.nd.empty((n,), dtype="float32")
f(x_tvm, y_tvm)  # 执行编译后的函数
print("x:", x_np)
print("y:", y_tvm.numpy(), " (期望 [2..9])")


# ============== 2. 二维向量加法 y[i][j] = x[i][j] + a ==============
print("\n===== 实验 1.2: 二维加法 y[i][j] = x[i][j] + a =====")
m, k = 4, 4
X = te.placeholder((m, k), name="X")
# ######################################################################
# AI 编译器和普通编程最核心的区别——你描述"结果长什么样"，而不是"怎么循环算出来"
# 循环由编译器生成，不是由你写。
# (m, k) 告诉 TVM："我要一个 m×k 的输出"。TVM 收到这个形状后，自动推导出要遍历所有位置，并且对每个 (i, j) 调用你的 lambda。
Y = te.compute((m, k), lambda i, j: X[i, j] + 1.0, name="Y")
# ######################################################################
s2 = te.create_schedule(Y.op)
f2 = tvm.build(s2, [X, Y], target="llvm")
X_np = np.arange(m * k, dtype="float32").reshape((m, k))
X_tvm = tvm.nd.array(X_np)
Y_tvm = tvm.nd.empty((m, k), dtype="float32")
f2(X_tvm, Y_tvm)
print("X:\n", X_np)
print("Y:\n", Y_tvm.numpy())


# ============ 3. 矩阵乘法 C[i][j] = sum_k A[i][k] * B[k][j] ============
print("\n===== 实验 1.3: 矩阵乘法 (MatMul) =====")

M, N, K = 4, 4, 4
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")

# ######################################################################
# 先创建一个"符号归约轴"：k 的范围是 [0, K)
k = te.reduce_axis((0, K), name="k")
# 再在 compute 里用这个符号
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# ######################################################################
# 调度： 分块（tile）：让内层循环更贴近cache
s3 = te.create_schedule(C.op)
# ######################################################################
# 第一步：先看编译器生成了什么循环：默认生成的循环（M=N=K=4）
# for (i = 0; i < 4; i++) {        // axis[0]：外层
#   for (j = 0; j < 4; j++) {      // axis[1]：中层
#     float acc = 0;
#     for (k = 0; k < 4; k++) {    // axis=k 的归约轴：内层
#       acc += A[i][k] * B[k][j];
#     }
#     C[i][j] = acc;
#   }
# }
# 上面代码的执行顺序是：(i=0,j=0) → (i=0,j=1) → (i=0,j=2) → (i=0,j=3) → (i=1,j=0) → ...

# 第二步：split 把一根轴"掰成两根"
# 把i和j轴各分成2块，i->(io, ii), j->(jo, ji)
io, ii = s3[C].split(C.op.axis[0], factor=2)  # 处理 i 轴
# 意思是：把 i 轴 [0..3] 拆成两层循环——外层 io 负责"第几块"，内层 ii 负责"块内第几个"：
# i 的范围 [0,1,2,3]  (4 个)
#     factor=2  →  每块 2 个 → 共 2 块

# io = i / 2   →   [0, 0, 1, 1]   ← 第几块
# ii = i % 2   →   [0, 1, 0, 1]   ← 块内第几个

# 同理 jo, ji = s3[C].split(axis[1], factor=2) 把 j 轴也拆成"块外 + 块内"。
jo, ji = s3[C].split(C.op.axis[1], factor=2)
# 重排顺序：先块外，再块内（io, jo, ii, ji）
# ######################################################################

# ######################################################################
# 第三步：reorder 调整这几层的先后顺序
# 把执行顺序从 io, ii, jo, ji 改成 io, jo, ii, ji——先跑完所有块，再跑块内：
# reorder 后：外层循环先遍历"块"，内层才遍历"块内元素"
# for (io = 0; io < 2; io++)
#   for (jo = 0; jo < 2; jo++)        // ← 先确定在哪个块
#     for (ii = 0; ii < 2; ii++)      // ← 再算块内
#       for (ji = 0; ji < 2; ji++)
#         C[io*2+ii][jo*2+ji] = ...;
s3[C].reorder(io, jo, ii, ji)
# 为什么这样改有好处？
# 把 4×4 的输出想象成 16 个格子，reorder 改变的是计算顺序：
# 默认 (i, j):           split+reorder 后 (io, jo, ii, ji):
# 按行扫：               按 2×2 块扫：
# j→                      每个块 2×2，先算完一块再下一块：
# (0,0)(0,1)(0,2)(0,3)                ┌─────────┬─────────┐
# (1,0)(1,1)(1,2)(1,3)                │ (0,0).. │ (0,2).. │   ← 块 0
# (2,0)(2,1)(2,2)(2,3)                │  ...    │  ...    │
# (3,0)(3,1)(3,2)(3,3)                ├─────────┼─────────┤
#                                     │ 块 2    │ 块 3    │
#                                     └─────────┴─────────┘
# 好处（数据局部性）：计算 (0,0) 时用到 A 的第 0 行；
# 算 (0,1) 还用 A 的第 0 行——CPU 缓存里已经有一份了，不用重新从内存读。
# 按块扫让相邻的计算复用同一批缓存行，内存访问更快。
# 这就是调度存在的意义：计算语义（算什么）完全不变，只是改变访问顺序（怎么算），换来性能。
# ######################################################################


f3 = tvm.build(s3, [A, B, C], target="llvm")

A_np = np.random.rand(M, K).astype("float32")
B_np = np.random.rand(K, N).astype("float32")
C_np = np.matmul(A_np, B_np)  # numpy 参考结果

A_tvm, B_tvm = tvm.nd.array(A_np), tvm.nd.array(B_np)
C_tvm = tvm.nd.empty((M, N), dtype="float32")
f3(A_tvm, B_tvm, C_tvm)

print("TVM 结果:\n", C_tvm.numpy())
print("numpy 参考:\n", C_np)
print("误差:", np.max(np.abs(C_tvm.numpy() - C_np)))

# ============ 4. 同一个计算，不同调度，结果一致 ============
print("\n===== 实验 1.4: 计算与调度分离 —— 不同调度结果一致 =====")


# 同样的 C = A@B，两种调度
def build_matmul(schedule_fn, tag):
    A_ = te.placeholder((M, K), name="A")
    B_ = te.placeholder((K, N), name="B")
    k_ = te.reduce_axis((0, K), name="k")  # ← 加这一行
    C_ = te.compute(
        (M, N), lambda i, j: te.sum(A_[i, k_] * B_[k_, j], axis=k_), name="C"
    )
    s_ = te.create_schedule(C_.op)
    schedule_fn(s_, C_)
    f_ = tvm.build(s_, [A_, B_, C_], target="llvm")
    C_out = tvm.nd.empty((M, N), dtype="float32")
    f_(tvm.nd.array(A_np), tvm.nd.array(B_np), C_out)
    return C_out.numpy()


# 调度 A: 朴素（默认）
def naive_sched(s, C):
    pass


# 调度 B: 向量化内层循环 (SIMD)
def vec_sched(s, C):
    s[C].vectorize(C.op.axis[1])


# 调度 C: 并行外层循环 (多核)
def par_sched(s, C):
    s[C].parallel(C.op.axis[0])


out_naive = build_matmul(naive_sched, "naive")
out_vec = build_matmul(vec_sched, "vectorize")
out_par = build_matmul(par_sched, "parallel")

print("naive     误差:", np.max(np.abs(out_naive - C_np)))
print("vectorize 误差:", np.max(np.abs(out_vec - C_np)))
print("parallel  误差:", np.max(np.abs(out_par - C_np)))
print("\n三种调度结果一致（计算与调度分离的正确性证明）")
