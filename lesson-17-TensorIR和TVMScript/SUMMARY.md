# Lesson 17: TensorIR / TVMScript 总结

## 课程目标



1. **用 TVMScript 写底层计算内核**——直接在 Python 里写循环，像写 C 一样控制计算
2. **理解"调度 = 循环变换"**——同一段数学，循环怎么写，性能差 37 倍
3. **掌握三把性能之刀**：循环重排（cache 友好）、`T.vectorized`（SIMD）、`T.parallel`（多核）
4. **把内核编译成 .so 部署到 C 程序**——打通"Python 写内核 → 动态库 → C 调用"完整链路
5. **理解 Relay 图和 te 表达式的层级关系**——为什么部署要走 `relay.build` 而非裸 `tvm.build`

---

## 一、三个实验回顾

| 实验 | 文件 | 内容 | 关键产出 |
|------|------|------|----------|
| 01 | `01_tvmscript_intro.py` | TVMScript 基本语法（三大件） | `@T.prim_func` / `T.Buffer` / `T.serial` |
| 02 | `02_tvmscript_matmul.py` | 手写 MatMul 四版本对比 | naive 6.45ms → opt_parallel 0.17ms（**37.7x**） |
| 03 | `03_tvmscript_deploy.py` + `main.c` + `Makefile` | 内核 → .so → C API 调用 | `relay.build` + graph_executor 三步流程 |

### 实验 1：TVMScript 基本语法

```python
@T.prim_func
def add_one(
    x: T.Buffer(8, "float32"),   # 输入：8 个 float32
    y: T.Buffer(8, "float32"),   # 输出：8 个 float32
):
    for i in T.serial(8):        # 串行循环
        y[i] = x[i] + 1.0

f = tvm.build(add_one, target="llvm")
```

**三大件**：
- `@T.prim_func` → 声明这是一个底层计算函数（TIR PrimFunc）
- `T.Buffer(形状, 类型)` → 声明输入输出张量（注意新版写法，`T.Buffer[(8,)]` 已废弃）
- `T.serial / T.vectorized / T.parallel / T.grid` → 循环控制

**与 te 的本质区别**：te 是"声明式"（描述算什么，编译器决定怎么算），TVMScript 是"命令式"（直接写怎么算，完全掌控循环）。两者编译产物等价，但 TVMScript 能看见并修改底层 IR。

**输出张量必须 `tvm.nd.empty` + `.numpy()` 取回**——`tvm.nd.array(y_np)` 会拷贝一份，结果写不回去。

### 实验 2：手写 MatMul —— 调度三把刀

四个版本，同一数学，性能天差地别：

| 版本 | 循环结构 | 耗时 | 相对 naive | 用到的优化 |
|------|----------|------|------------|------------|
| naive | i → j → k | ~6.5 ms | 1.00x | 无 |
| opt | i → k → j + vectorize | ~0.87 ms | **7.5x** | 重排 + SIMD |
| parallel | i 并行，但 k 最内 | ~3.6 ms | 1.8x | 只并行，没修 cache |
| **opt_parallel** | i 并行 + k 中间 + j 向量化 | **~0.17 ms** | **37.7x** | 三刀全上 |

**三把刀**：

1. **循环重排 → cache 友好**
   - naive：k 在最内层，`B[k][j]` 随 k 变化跨行跳跃，cache line 用 1/16（浪费 16 倍带宽）
   - opt：k 放中间、j 放最内层，`B[k][:]` 和 `C[i][:]` 连续访问用满 cache line
   - `A[i][k]` 是标量 → 加载一次进 SIMD 寄存器**广播复用 256 次**，零内存访问（rank-1 update / 外积累加）

2. **`T.vectorized` → SIMD 指令**
   - `T.serial` 每次迭代算 1 个元素（标量），`T.vectorized` 一次算 4/8/16 个（AVX/NEON）
   - 前提：循环体逐元素独立（无数据依赖、无分支）
   - k 循环是累加轴有依赖，**不能**向量化；重排后内层 j 独立，**可以**向量化

3. **`T.parallel` → 多核并行**
   - 前提：迭代间无写冲突（i 并行写不同行 C[i,:]，安全；k 是累加轴不能并行）
   - ⚠️ **并行不解决 cache 问题，反而放大**——16 核抢带宽跑低效循环 = 灾难

**关键教训**（真实工业界会踩）：
- 容差要用 `rtol=1e-4` 而非默认 `1e-7`——不同累加顺序的浮点误差会误报失败
- 写 `+=` 前必须先清零（buffer 是裸内存）

### 实验 3：部署到 C —— 为什么必须走 relay.build

**完整链路**：算子表达式 → Relay IR → TIR → LLVM 机器码 → `.so` → C 调用

```python
# ① Relay 算子构造 IRModule（手写"图"）
A = relay.var("A", shape=(M, K), dtype="float32")
B = relay.var("B", shape=(K, N), dtype="float32")
C = relay.nn.matmul(A, B)
mod = tvm.IRModule.from_expr(relay.Function([A, B], C))

# ② relay.build 编译（和 Lesson 15/16 完全一致）
lib = relay.build(mod, target="llvm")
lib.export_library("./libmatmul.so")
```

C 端调用流程与 Lesson 16 **完全一致**：`TVMModLoadFromFile` → `default` 工厂 → `set_input("A"/"B")` → `run` → `get_output(0)`。


## 二、核心概念速查

### T.serial vs T.vectorized vs T.parallel

```
T.serial       → 每次迭代算 1 个元素 → 标量指令
T.vectorized   → 每次迭代算 4/8/16 个 → SIMD 指令（x86 AVX / ARM NEON）
T.parallel     → 迭代分配给多线程 → 多核并行
```

注意：这些是"循环的注解"，不是不同类型的循环。`T.vectorized` 只对最内层有意义，且要求迭代间无依赖。

### 能向量化吗？

| 循环体 | 能向量化吗 | 原因 |
|--------|-----------|------|
| `C[i,j] = A[i,k] * B[k,j]` | ✅ | 每个 j 独立 |
| `s += A[i,k]`（k 循环） | ❌ | 数据依赖 |
| 有 if/break 的分支 | ❌ | 控制流不统一 |

### 矩阵乘的最优循环结构

```
for i in parallel(M):      # ① 并行：每行一个线程（写不同行，无冲突）
    for k in serial(K):    # ② k 中间层：A[i,k] 广播复用
        for j in vectorized(N):  # ③ j 最内层：B/C 连续 + SIMD
            C[i, j] += A[i, k] * B[k, j]
```

### Relay 图 vs te 表达式（实验 3 的核心疑问）

```
relay.Function([A, B], C)   ← 高层 IR（图），描述"用什么算子"
relay.nn.matmul(A, B)       ← 高层算子（黑盒），内部隐藏了一个 te.compute

te.compute(...)             ← 低层 DSL，描述"怎么算"（可调度循环）
```

两者**不能直接替换**：`relay.nn.matmul` 内部隐藏了 TVM 预置的 te 实现；你写的 te.compute 是自定义实现。层级不同，一个在 Relay 图层，一个在 te 表达式层。