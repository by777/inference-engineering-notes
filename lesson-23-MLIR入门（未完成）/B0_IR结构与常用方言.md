# Lesson 23-B：IR 结构与常用方言

## 一、方言（Dialect）

**方言 = 一套"词表"，每个方言管一类操作。** 就像 C 标准库拆成 `<math.h>`（数学函数）、`<string.h>`（字符串函数），MLIR 把操作按"管什么"拆成一个个方言。

`arith` 和 `scf` 就是其中两个：一个管"计算"，一个管"控制流"。

**为什么叫"方言"而不叫"库"**：因为它们**不是函数，是操作（operation）**——它们是编译器 IR 图里的节点，可以被 pass 优化、被降低（lower）。比如 `scf.for` 可以被 `convert-scf-to-cf` 这个 pass 整个替换成一组跳转指令。

**方言的意义**：MLIR 的"词表"是**开放的**——你觉得 `arith` 不够用，可以定义自己的方言（比如 `npu.mac` 表示你 NPU 的乘加指令），然后写 pass 把 `arith`/`scf` 降到你自己的方言。这是芯片公司干的事。

---

## 二、IR 结构：一棵树（B1 实验）

**`.mlir` 不是"代码"，是"图"的文字描述**。MLIR 在内存里维护一张树形结构，文本只是打印出来的样子。

### 四层结构（从外到内）

```
module                          ← 第1层：整张图（编译单元）
  └─ func.func @main            ← 第2层：函数（本身是一个 Region）
       └─ Block（函数体）         ← 第3层：顺序语句序列
            └─ scf.if { } else { }  ← 第4层：嵌套 Region（then/else 各一个）
```

| 层级 | 是什么 | 类比 |
|---|---|---|
| `module { }` | 整张图的边界 | ONNX 文件 / C 的 `.c` |
| `func.func @main` | 函数 | `int main()` |
| Block | 顺序执行的语句序列 | C 函数体 |
| Region | "块"的容器，**可嵌套** | C 的 `{}` 代码块 / ONNX subgraph |

**核心：Region 可以嵌套**——函数体是一个 Region，`scf.if` 的 then/else 各是一个 Region，Region 里又是 Block。这就是"一棵树"的含义，不是平的列表。

**SSA 命名是局部的**：两个 Region 里可以都用 `%t`/`%e` 互不干扰（各自作用域），就像 C 里两个 if 块各自声明 `int x`。

**`scf.yield`**：Region 不能 `return` 给外层，靠 `scf.yield %x` 把值"送出去"（像 return，但给外层 Region 用）。

---

## 三、`arith` = arithmetic（算术）

管**纯计算**的操作：加减乘除、比较、类型转换。

| 写法 | 全称 | C 类比 |
|---|---|---|
| `arith.constant` | 常量 | 字面量 `1` |
| `arith.addi` | add integer（整数加） | `+` |
| `arith.subi` | subtract integer（整数减） | `-` |
| `arith.muli` | multiply integer（整数乘） | `*` |
| `arith.divsi` | divide signed integer（有符号整数除） | `/` |
| `arith.addf` | add float（浮点加） | `+`（float） |
| `arith.cmpi` | compare integer（整数比较） | `==`、`<` |
| `arith.fptosi` | float to signed int（浮点转整型） | `(int)` 强转 |

**命名规律**（很有用）：
- 末尾 `i` = integer（整数），如 `addi`
- 末尾 `f` = float（浮点），如 `addf`
- 开头 `sub`/`mul`/`div`/`rem` = 减/乘/除/余
- `s` 出现 = signed（有符号），如 `divsi`

**为什么需要分 i/f？** 因为 MLIR 的类型系统里 `i32` 和 `f32` 是不同类型，加法的语义不一样（整数加法取模、浮点加法有舍入），所以分成 `addi`/`addf` 两个操作，编译器好校验、好优化。

**一句话**：`arith` 管"算"，不管"跳"。

---

## 四、`scf` = Structured Control Flow（结构化控制流）

管**程序的控制结构**：循环、条件分支。

| 写法 | 全称 | C 类比 |
|---|---|---|
| `scf.if` | if（条件） | `if (cond) { } else { }` |
| `scf.for` | for（循环） | `for (i = lb; i < ub; i += step)` |
| `scf.while` | while（循环） | `while (cond)` |
| `scf.yield` | 把值交出去 | `return`（给外层 Region） |

**"Structured"（结构化）什么意思？**——关键概念。

结构化 = 控制流是**嵌套的、有明确入口出口的**，就像 C 的 `if`/`for`。

对比它的反面——**非结构化控制流**（`cf` 方言，control flow）：

```mlir
cf.br ^label          // 无条件跳转（goto）
cf.cond_br %c, ^a, ^b // 条件跳转（if-goto）
^label: ...
```

这就是**汇编的跳转**，可以跳去任何地方，没有结构。

```
scf（结构化）:  scf.for { ... } → 循环体 → scf.for 结束     （有清晰入口出口）
cf（非结构化）:  cf.br ^bb2 → ^bb1: ... → ^bb2: ...         （goto 到处跳）
```

**一句话**：`scf` 是"长得像 C 的高级控制流"（好读好优化），`cf` 是"长得像汇编的低级控制流"（贴近机器）。

### scf.for 与 iter_args（B4 核心概念）

```mlir
%result = scf.for %i = %lb to %ub step %step
    iter_args(%acc = %init) -> (i32) {   // 循环携带值
  %i32 = arith.index_cast %i : index to i32
  %new = arith.addi %acc, %i32 : i32
  scf.yield %new : i32                    // 传给下一轮
}
```

对应 C：
```c
int acc = init;                 // iter_args 初始化
for (i = lb; i < ub; i += step) {
  int new = acc + i;            // 循环体
  acc = new;                    // scf.yield
}
return acc;                     // %result
```

**为什么要 iter_args？** 因为 **SSA 不可变**——不能写 `acc += i`（acc 被改了）。所以只能"每轮产生新 acc 传给下一轮"。`iter_args` 就是 SSA 世界里的循环变量写法。

---

## 五、`tensor` = 值语义（不可变张量）

管**张量数据**。核心认知：**tensor 是"算的值"，不可变**——对 tensor 的任何操作都产生新 tensor，原 tensor 不变。类比 ONNX 图里流动的 tensor。

| 写法 | 含义 | C 类比 |
|---|---|---|
| `tensor<2x3xf32>` | 类型：2行3列 float32 | `float[2][3]` |
| `dense<[[...], [...]]>` | 完整数据（密集） | 数组字面量 |
| `tensor.extract %t[%i, %j]` | 读元素（不修改） | `t[i][j]` |
| `tensor.generate { ^bb0(%a,%b) ... }` | 按位置生成新 tensor | 双层循环填充 |
| `tensor.yield %x` | 当前元素的值交出去 | 循环体里赋值 |

**坑：索引必须是 SSA 值**——不能写 `tensor.extract %t[0, 1]`（静态索引不行），必须先 `%i = arith.constant 0 : index` 再用 `%t[%i, %j]`。

**坑：`: tensor<2x3xf32>` 是操作数类型，不是结果类型**——`tensor.extract %t[%i, %j] : tensor<2x3xf32>` 里这个类型标注的是 `%t`（操作数），`%e` 的结果是 `f32` 标量（从 2x3 里取一个元素）。结果类型能推断所以省略了。验证：`mlir-opt --mlir-print-op-generic` 能看到完整的 `(tensor<2x3xf32>, index, index) -> f32`。

**tensor.generate 为什么是声明式**：它不写循环，只写"每个位置怎么算"（Region 里 extract 原值、加 1、yield），遍历由操作本身提供。编译器知道各位置互不依赖 → 可并行、可向量化、可在 NPU 上调度。这是 AI 芯片编译器的基础。

**为什么不可变重要**：`%t` 永远不变 → 编译器可以放心优化（重排/融合/共享），不用担心"某处偷偷改了它"。代价是每次操作产生新 tensor（费内存）→ 这就引出 memref（B3）和 bufferization（后面课程）：**计算阶段用 tensor 方便优化，落内存时转 memref 省空间**。

---

## 六、`memref` = 内存语义（可变缓冲区）

管**内存**。核心认知：**memref 是"放的内存"，可读可写**——`store` 修改原内存。类比 **此前 NPU 的片上特征/权重缓冲**（一块可读写的片上内存）。

| 写法 | 含义 | C 类比 |
|---|---|---|
| `memref<2x3xi32>` | 类型：2行3列 int32 | `int m[2][3]` |
| `memref.alloc()` | 分配内存 | `malloc` / 在片上缓冲划一块区域 |
| `memref.store %v, %m[%i, %j]` | 写入（修改原内存！） | `m[i][j] = v` |
| `memref.load %m[%i, %j]` | 读取 | `v = m[i][j]` |
| `memref.dealloc %m` | 释放内存 | `free(m)` |

**生命周期**：`alloc` 和 `dealloc` 必须配对——分配不释放就是**内存泄漏**。

**内存布局**：`memref<2x3xi32>` 按行连续排列（row-major，和 C 一样）：`[0][0]` 在首地址（偏移 0），`[0][1]` 偏移 4，`[1][2]` 偏移 20。二维索引 `[行][列]` 对应偏移，和 C 完全一致。

**为什么 alloc 的是"引用"不是"值"**：`%m` 不包含数据本身，是指向一块存储空间的句柄。就像 C 的指针/数组名。

### 本课最重要：tensor vs memref 对比

| | tensor（B2） | memref（B3） |
|---|---|---|
| 语义 | **值**（不可变） | **内存**（可变） |
| 写 | 无写操作，只能造新值 | `memref.store` 原地改 |
| 读 | `tensor.extract` | `memref.load` |
| 类比 | ONNX 图里流动的 tensor | **此前 NPU 的片上特征/权重缓冲** |
| 生命周期 | 无（值自生自灭） | `alloc` / `dealloc` 显式管理 |
| 编译器视角 | 高级、可优化、可融合 | 低级、贴近硬件、有地址 |

**一句话记忆**：
> **tensor 是"算的值"，memref 是"放的内存"。** 计算图里流动的是 tensor，落到硬件上占的是 memref。

**为什么这个对比对 AI 芯片最重要**：整个编译流程就是
```
tensor（计算阶段，方便优化）
   ↓  bufferization（后面课程的核心！）
memref（落地阶段，占内存，可分配地址）
```
你在 此前 NPU 做的片上特征/权重缓冲地址规划，就是"给 memref 分配地址"；模型计算图里流动的 tensor 就是"还没落地的值"。**bufferization 就是把你此前 NPU 手工做的"规划内存"自动化**。

---

## 七、lower：从高级方言到 LLVM（B4 实验）

### 7.1 为什么不能直接执行

`scf`/`arith`/`tensor` 这些高级方言是"给人看的图"，**CPU 不认识**。必须一步步翻译成 CPU 认识的语言——这个过程就是 **lower（降级）**。

### 7.2 scf.for 降到 LLVM（对应汇编循环骨架）

```
scf.for（高级，结构化）        →   llvm.br ^bb1 / llvm.icmp / llvm.cond_br（低级，跳转）
```

`scf.for` lower 后变成经典汇编循环骨架：

```llvm
llvm.br ^bb1(%0, %3)          // 初始化：跳循环头，带 i 和 acc 初值
^bb1:                         // 循环头
  llvm.icmp "slt" %4, %1      //   i < 4 ?
  llvm.cond_br %6, ^bb2, ^bb3 //   是→循环体，否→退出
^bb2:                         // 循环体
  llvm.add %5, %7             //   acc = acc + i
  llvm.br ^bb1(%9, %8)        //   i++，跳回循环头
^bb3:                         // 出口
  llvm.return %5
```

**这就是你此前 NPU 写过的汇编循环**——比较、条件跳、累加、更新跳回。MLIR 用 `convert-scf-to-cf` 这个 pass 自动从 `scf.for` 生成这套骨架。

**分工**（B4 实验的 scf.for 里看得很清楚）：
- `scf.for` 决定"循环怎么走"（结构）
- `arith.addi` 决定"每轮算什么"（计算）
- **scf 搭骨架，arith 填肉**

### 7.3 编出来的 LLVM IR 是什么？—— 中间产物，不是废品

`mlir-opt` 打印的 LLVM IR 看起来"用不了"，但它有两个去向：

| 去向 | 说明 |
|---|---|
| **JIT 执行** | `mlir-cpu-runner` 内部用 LLVM JIT 把它当场编译成机器码执行（B4 输出 6 就是证明） |
| **LLVM 后端** | 继续交给 `llc` 等生成汇编 → 目标文件 → 可执行文件 |

类比 C 编译流程：
```
main.c → main.s（汇编，中间产物）→ main.o → a.out
B4_scf_lower.mlir → LLVM IR（中间产物）→ 机器码
```

**MLIR 的价值恰恰在这个中间层**：编译器不能一步从高级语言跳到机器码，中间必须有"能检查、能优化、能转换"的表示。你能亲眼看到（常量折叠、for→跳转），才能理解和修改——这就是"编译器"和"黑盒工具"的区别。

### 7.4 JIT 是什么

**JIT = Just-In-Time，即时编译——"用到才编译"**。

对比两种编译时机：

| 方式 | 什么时候编译 | 例子 |
|---|---|---|
| **AOT**（Ahead-Of-Time，预编译） | 运行**前**一次性编好 | `gcc main.c -o main`，编完存成可执行文件 |
| **JIT**（Just-In-Time，即时编译） | 运行**时**边跑边编 | `mlir-cpu-runner` 当场把 LLVM IR 编译成机器码执行 |

**MLIR 里 JIT 在哪**：就是 `mlir-cpu-runner` 干的事——把 `mlir-opt` 输出的 LLVM IR 当场编译成机器码执行，B4 输出的 `6` 就是 JIT 跑出来的。

**为什么实验用 JIT 而不是 AOT**：

| | JIT（实验用） | AOT（产品用） |
|---|---|---|
| 修改后 | 直接重跑，即时生效 | 要重新编译 |
| 调试 | 能看 IR 中间过程 | 看不到 |
| 用途 | 学习、验证、快速原型 | 部署、性能 |

**注意**：真实产品（QNN、寒武纪工具链）部署时多用 **AOT**——模型先编译成 `.bin`/`.so` 再拷到板子上跑（像 流式降噪 的 `003_fp16_ctx.bin`）。JIT 是学习阶段用来快速验证的。

### 7.5 MLIR 完整链路（从 .mlir 到跑起来）

把 B 组所有实验串起来，完整链路：

```
高级方言 IR（scf/arith/tensor/func，B1~B3 手写）
   ↓ ① pass 优化（canonicalize / 融合）
优化后的 IR
   ↓ ② lower pass 链（convert-scf-to-cf / convert-arith-to-llvm / convert-func-to-llvm）
LLVM IR（B4 step1 看到的）
   ↓ ③ JIT 编译（mlir-cpu-runner）
机器码
   ↓ ④ 执行
结果（输出 6）
```

| 环节 | 工具 | 干什么 | 你看到的 |
|---|---|---|---|
| ① 高级 IR | 你手写 | 描述"算什么"（图） | `scf.for`、`arith.addi` |
| ② 优化 pass | `mlir-opt --canonicalize` | 让图更简单 | `7+5`→`12` |
| ③ lower pass | `mlir-opt -pass-pipeline=...` | 翻译成低级方言 | `scf.for`→跳转 |
| ④ LLVM IR | `mlir-opt` 输出 | 中间表示（贴近机器但可读） | `llvm.br`/`llvm.icmp` |
| ⑤ JIT | `mlir-cpu-runner` | 运行时编译成机器码 | 输出 `6` |

**和你的此前 NPU/TVM 对照**：
- 你在 此前 NPU 写的"宏+脚本翻译 asm" = ③ lower pass（把高级翻译成低级）
- TVM 的 `tvm.build(target='llvm')` = ③④（TVM 也是降到 LLVM）
- TVM 部署到板子用 AOT（编好 .so 拷过去），MLIR 实验用 JIT（运行时编）