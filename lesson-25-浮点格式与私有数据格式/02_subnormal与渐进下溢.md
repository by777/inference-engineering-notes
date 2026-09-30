# 02 subnormal 与渐进下溢（gradual underflow）

> **一句话**：指数字段只有那么宽，最小正规数之下"没有档位了"。
> IEEE754 的解法是**征用 $E=0$ 这一档、并把隐含前导 1 换成 0**，让表示范围继续往下延伸——
> 代价是**有效位数线性退化**。这叫用精度换连续性。

---

## 0. 问题的来源

以 fp16 为例（`1+5+10`，bias = 15），指数字段 $E$ 是 5 位，取值 $0 \sim 31$，但两头被征用：

| $E$ 字段 | 用途 |
|---|---|
| `11111` (31) | **inf / NaN** |
| `00000` (0) | ？ |
| `00001` ~ `11110` (1~30) | 正规数 |

所以正规数的最小值是 $E = 1, M = 0$：

$$
x_{\min} = 2^{1 - 15} = 2^{-14} \approx 6.1035 \times 10^{-5}
$$

**于是出现一个空洞**：从 $2^{-14}$ 往下到 0，这一大片怎么表示？

### 如果什么都不做：abrupt underflow（突降为零）

那么所有小于 $2^{-14}$ 的数都变成 0。后果：

| 运算 | 数学结果 | 若无 subnormal |
|---|---|---|
| $2^{-8} \times 2^{-8}$ | $2^{-16} = 1.53\times10^{-5}$ | **0** ← 两个精确的正规数相乘，结果成了 0 |
| $2^{-6} \times 2^{-9}$ | $2^{-15} = 3.05\times10^{-5}$ | **0** |
| $2^{-12} \times 2^{-12}$ | $2^{-24} = 5.96\times10^{-8}$ | **0** |

**乘法链会莫名其妙塌成 0**。这在量化极其常见——$s_w \times s_x$ 就是两个小 scale 相乘。

### 解法：征用 $E = 0$

把 $E=0$ 定义为 **subnormal（非规格化数 / 次正规数）**：

$$
x_{\text{sub}} = (-1)^{s} \times 2^{\,1 - \text{bias}} \times \left(\boxed{0} + \frac{M}{2^{m}}\right)
$$

**和正规数的唯一差别**：隐含的前导位从 $1$ 变成 $0$。

| | 正规数 | subnormal |
|---|---|---|
| $E$ 字段 | $[1,\,2^{e}-2]$ | $0$ |
| 隐含前导位 | **1** | **0** |
| 间距（ulp） | $2^{E-\text{bias}-m}$（随指数变化） | $2^{1-\text{bias}-m}$（**固定**） |
| 相对精度 | 恒定 $2^{-(m+1)}$ | **随值变小而线性下降** |
| 特殊成员 | — | $M=0$ → **±0** |

**间距固定**这一点是关键：指数不能再减了，只好让间距停在最小正规数的 ulp 上。

---

## 1. 实测：边界是严丝合缝的

```
--- 边界：位模式只差最后一位，值却跨过 2^-14 ---
  最小正规数     0 00001 0000000000  -> 6.1035e-05
  最大 subnormal 0 00000 1111111111  -> 6.0976e-05
  最大 subnormal + ulp == 最小正规数 ? True
  -> 间距固定 = 5.9605e-08 = 2^-24，无缝衔接（gradual underflow）
```

两条位模式**只差最后一位**，但左边界跨过了 $2^{-14}$：

| | 位模式 | 值 |
|---|---|---|
| 最小正规数 | `0 00001 0000000000` | $6.1035\times10^{-5}$ |
| 最大 subnormal | `0 00000 1111111111` | $6.0976\times10^{-5}$ |

**"渐进"（gradual）这个词就是从这里来的**：数值往下走时不是悬崖式掉到 0，而是一位一位地让出精度。

---

## 2. 代价：有效位数线性退化

subnormal 区里 $E$ 恒为 0，**能变的只有 $M$**，所以：

| $M$ | 需要的位数 | 值 | 相对精度 |
|---|---|---|---|
| 1023 | 10 | 6.0976e-05 | **10 bit** |
| 512 | 10 | 3.0518e-05 | 10 bit |
| 256 | 9 | 1.5259e-05 | 9 bit |
| 128 | 8 | 7.6294e-06 | 8 bit |
| 64 | 7 | 3.8147e-06 | 7 bit |
| 32 | 6 | 1.9073e-06 | 6 bit |
| 16 | 5 | 9.5367e-07 | 5 bit |
| 8 | 4 | 4.7684e-07 | 4 bit |
| 4 | 3 | 2.3842e-07 | 3 bit |
| 2 | 2 | 1.1921e-07 | 2 bit |
| 1 | 1 | 5.9605e-08 | **1 bit** |

$$
\text{精度} \approx \log_2 M \quad\text{（而}\ M = x / 2^{1-\text{bias}-m}\text{）}
$$

**注意最后一行**：$M = 1$ 时只有 1 位精度，相对误差高达 **50%**。所以 subnormal 只是"能用"，不是"好用"。

> 这也解释了为什么 subnormal 让私有格式的**收益可以被精确量化**（见第 03 章）：
> re-bias 能救回的位数，正好是 subnormal 折叠掉的那些位。

---

## 3. fp32 也有 subnormal（而且 bf16 也有）

subnormal **不是 fp16 的缺陷**，而是整个 IEEE754 家族的通用机制：

| 格式 | 偏置 | 尾数位 | 最小正规数 | subnormal 间距 | subnormal 个数 |
|---|---|---|---|---|---|
| fp32 | 127 | 23 | 1.1755e-38 | 1.4013e-45 | **8388607** |
| bf16 | 127 | 7 | 1.1755e-38 | 9.1835e-41 | **127** |
| fp16 | 15 | 10 | 6.1035e-05 | 5.9605e-08 | **1023** |

$$
\boxed{\text{subnormal 个数} = 2^{m} - 1} \qquad\qquad \text{subnormal 间距} = 2^{\,1-\text{bias}-m}
$$

| 格式 | 间距推导 | 结果 |
|---|---|---|
| fp16 | $2^{1-15-10}$ | $2^{-24}$ |
| fp32 | $2^{1-127-23}$ | $2^{-149}$ |
| bf16 | $2^{1-127-7}$ | $2^{-133}$ |

**三个格式都有，机制完全一样。** 区别只是 fp16 因为尾数少、指数字段窄，subnormal 占的比例大得多。

---

## 4. ⚠️ FTZ：subnormal **默认就是开的**，风险是"被关掉"

这是本课对你**实际工作影响最大**的一条。

> **结论先给**（实测见 [第 7 节](#7-编译选项与-ftz-控制实测)）：
> **不需要为 subnormal 开启任何编译选项** —— IEEE754 合规是**默认状态**，
> `-O0/-O1/-O2/-O3` 下 subnormal 都正常。
>
> 真正要防的是**被优化选项关掉**：`-ffast-math`（具体触发者是其中的
> `-funsafe-math-optimizations`）会打开 FTZ/DAZ，subnormal 直接变 0。
>
> $$ \text{问题不是"怎么开"，而是"别被关掉"} $$
>
> 一句话记法：**默认安全，`fast-math` 危险，`-ffinite-math-only` 无关。**

### 为什么硬件会砍掉 subnormal

subnormal 没有隐含前导 1，硬件要额外做：

- **leading-zero 计数**（数前导零才能重新规格化）
- **结果重新规格化**（结果可能从正规掉进 subnormal，需要额外的移位通路）

为了省这套逻辑和功耗，不少 **NPU / DSP / GPU（在 fast-math 模式下）** 选择直接 **FTZ（flush-to-zero）**：**只要指数域全 0，就当 0 处理**。

### 模拟对比（`float_bits.py`，算法模型，非真实硬件）

> ⚠️ 下面是**算法模型**里开/关 subnormal 的差别，用来理解语义。
> **真实编译器和硬件的实测见 [第 7 节](#7-编译选项与-ftz-控制实测)。**

```
--- FTZ 风险（模型演示）：不支持 subnormal 时会被刷成 0 ---
  输入 1e-05  IEEE(有subnormal) ->   1.0014e-05    FTZ(刷0) ->            0
  输入 1e-07  IEEE(有subnormal) ->   1.1921e-07    FTZ(刷0) ->            0
  输入 1e-08  IEEE(有subnormal) ->            0    FTZ(刷0) ->            0
```

| 输入 | IEEE 结果 | FTZ 结果 | 差别 |
|---|---|---|---|
| $10^{-5}$ | $1.0014\times10^{-5}$（subnormal，5 位精度） | **0** | 数值消失 |
| $10^{-7}$ | $1.1921\times10^{-7}$（subnormal，4 位精度） | **0** | 数值消失 |
| $10^{-8}$ | 0（低于 $2^{-24}$，IEEE 也是 0） | 0 | 一致 |

### 对量化部署的直接影响

$$
\text{若 } s_w \cdot s_x < 2^{-14} \;\Longrightarrow\; \begin{cases} \text{软件/模拟器（IEEE）} & \text{精度下降但能跑} \\ \text{板子（FTZ）} & \textbf{数值变 0，整层输出全废} \end{cases}
$$

**这是"模拟器结果好、板子结果崩"的经典来源之一。**

> 实践检查项（按"能不能自己查"排序）：
> 1. 量化校准算出的 scale 里，**最小的那个是多少**？有没有低到 $s_w \cdot s_x < 2^{-14}$？
>    → **不用等板子**，校准完扫一遍 scale 分布就能提前发现
> 2. **编译选项里有没有 `-ffast-math` / `-Ofast` / `-funsafe-math-optimizations`**？
>    → 实测：默认**不开**；上面这三个会打开 FTZ/DAZ。注意 `-ffinite-math-only` 单独用**不影响** subnormal
> 3. 你的模拟器是按 IEEE 处理 subnormal，还是也模拟 FTZ？
> 4. 目标硬件的 FP 单元实际 FTZ 位是几？
>    → 跑本课 `ftz_probe.c`，它直接读 `MXCSR`（x86）/ `FPCR`（AArch64）告诉你
>
> 检查命令与完整实测矩阵见 **[第 7 节](#7-编译选项与-ftz-控制实测)**。

---

## 5. 为什么值得要这个设计：连续性 vs 精度

| 方案 | 精度 | 连续性 |
|---|---|---|
| **abrupt underflow**（刷 0） | 0 位 | ✗ 有悬崖 |
| **subnormal / gradual** | 1~10 位（线性退化） | ✓ 连续 |
| **re-bias**（第 03 章） | 11 位（满） | ✓ 连续 |

subnormal 买到的是**数值连续性**：$x - y$（$x \approx y$）或 $x \times y$ 不会突然变成 0，而是给出一个精度降低但**方向正确**的结果。

**但请注意它的定位**：subnormal 是"把灾难降级成精度损失"的**半个补丁**。当 scale 这种量级小、又对精度敏感的量落进来时，你仍然需要第 03 章的 **re-bias**。

---

## 6. 特殊值速查

| $E$ | $M$ | 含义 |
|---|---|---|
| 0 | 0 | **±0**（符号位有效，所以有 $-0$） |
| 0 | ≠0 | **subnormal** ← 本章主角 |
| 1 ~ $2^e-2$ | 任意 | **正规数** |
| 全 1 | 0 | **±inf** |
| 全 1 | ≠0 | **NaN**（$M$ 最高位 1 = quiet，0 = signaling） |

> 📖 **NaN 的完整内容**（65536 码位分配、quiet/signaling 区分、三个实践坑、E4M3 为何没有 inf）
> 在 **[01 章第 1.2–1.3 节](01_浮点数的解剖.md)** —— 那里是"特殊码位"的主场，本文不重复。

**顺带一个设计观察**：$E$ 全 0 同时承担"零"和"subnormal"，$E$ 全 1 同时承担"inf"和"NaN"——
**IEEE754 把每个字段的边界值都榨干了**。这种"一个码位干两件事"的手法，在私有格式里会反复出现。

---

## 7. 编译选项与 FTZ 控制（实测）

### 7.1 结论：**不需要为 subnormal 开启任何选项；要担心的是被关掉**

很多人以为要"开启 subnormal"——**恰恰相反**：
IEEE 754 合规是默认状态，**subnormal 默认就是开的**。
真正的问题是**某些优化选项会把它关掉**。

### 7.2 实测矩阵

用 `ftz_probe.c` 实测（x86_64 gcc 11.4 / clang 14；aarch64 gcc 11.4 经 qemu 验证，**结论一致**）：

| 编译选项 | FTZ | DAZ | `1e-40` 存活？ |
|---|---|---|---|
| `-O0` / `-O2` / `-O3` | 0 | 0 | ✅ **subnormal 正常** |
| **`-ffast-math`** | **1** | **1** | ❌ **变 0** |
| **`-funsafe-math-optimizations`** | **1** | **1** | ❌ **变 0** |
| `-ffinite-math-only` | 0 | 0 | ✅ 正常（**不影响** subnormal） |
| `-fno-fast-math` | 0 | 0 | ✅ 正常 |

**两条关键结论**：

1. **触发 `-ffast-math` 里 FTZ 的成员是 `-funsafe-math-optimizations`**，
   不是 `-ffinite-math-only`（尽管 `-ffinite-math-only` 单独会禁掉 inf/NaN 假设）。
   实测 `-ffast-math -fno-finite-math-only` **仍然** FTZ=1，证实了这一点。

2. **`-ffast-math` 还顺带打开了 `-fno-signed-zeros`、`-fassociative-math`（允许重结合）**。
   对**累加顺序敏感**的计算（量化累加、Kahan 求和）都可能因此改变结果 —— 不只是 subnormal 的问题。

### 7.3 硬件层：寄存器在哪

| 架构 | 寄存器 | 位 | 语义 |
|---|---|---|---|
| **x86** | `MXCSR` | bit 15 | **FTZ**（结果侧刷零） |
| **x86** | `MXCSR` | bit 6 | **DAZ**（输入侧当零） |
| **AArch64** | `FPCR` | bit 24 | **FZ**（flush-to-zero） |
| **AArch64** | `FPCR` | bit 19 | `FZ16`（只影响 fp16 运算） |

实测值：正常编译 `MXCSR=0x00001FB2`、`FPCR=0x0000000000000000`；
`-ffast-math` 后 `MXCSR` 的 bit15/bit6 变成 1、`FPCR=0x0000000001000000`（bit24 置 1）。

### 7.4 ⚠️ DAZ 与 FTZ 不是一回事（实测演示）

这两个常被混为一谈，但语义不同：

| 机制 | 作用位置 | 效果 |
|---|---|---|
| **DAZ** | **输入侧** | subnormal 操作数被 FPU **读入**时当作 0 |
| **FTZ** | **结果侧** | 运算**结果**是 subnormal 时刷成 0 |

实测演示（`ftz_probe.c` 第 2 节，用 `in + in` 而非 `in + 0.0f`）：

| 编译 | 1e-40 的位模式（内存） | `in + in` 的结果 |
|---|---|---|
| `-O2` | `0x000116C2` SUBNORMAL | `0x00022D84` = 2e-40 ✅ |
| `-ffast-math` | `0x000116C2` SUBNORMAL | **`0x00000000`** ❌ |

**注意第一行那个漂亮的现象**：`-ffast-math` 下位模式还在内存里（`memcpy` 不经 FPU），
但一旦**被 FPU 读入**（转 double 打印、参与运算），立刻变成 0 —— **这就是 DAZ**。

> ⚠️ **写检测代码的坑**：`x + 0.0f` 这种恒等运算会被 `-ffast-math`（含 `-fno-signed-zeros`）
> 优化掉，测不到东西。要用 `in + in` 这类**真运算**。

### 7.5 运行时补救：即使编译时被关了，也能救回来

如果你**必须**用 `-ffast-math` 换性能（比如某些 kernel），可以在**进程启动时清一次**：

```c
/* x86：清 MXCSR 的 FTZ(bit15) 和 DAZ(bit6) */
unsigned csr = _mm_getcsr();
csr &= ~((1u << 15) | (1u << 6));
_mm_setcsr(csr);

/* AArch64：清 FPCR 的 FZ(bit24) 和 FZ16(bit19) */
unsigned long fpcr;
__asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
fpcr &= ~((1UL << 24) | (1UL << 19));
__asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
```

实测（`./ftz_probe --ieee`）：

| 编译 | 参数 | FTZ/FZ | 1e-40 结果 |
|---|---|---|---|
| `gcc -O2 -ffast-math` | 无 | 1 | 0 ❌ |
| `gcc -O2 -ffast-math` | `--ieee` | **0** | **`0x000116C2` SUBNORMAL ✅** |
| `aarch64 -O2 -ffast-math` | 无 | FZ=1 | 0 ❌ |
| `aarch64 -O2 -ffast-math` | `--ieee` | **FZ=0** | **subnormal ✅** |

**注意局限**：这只是救回**硬件状态位**。
`-ffast-math` 编译期做的其他假设（重结合、无 NaN、无 signed zero）**救不回来** ——
那些是编译器在代码生成时就基于假设优化掉的。所以运行时清位是**补救**，不是**等价于不用 fast-math**。

### 7.6 怎么检查自己的环境（三步）

```bash
# 1. 查编译选项里有没有 fast-math
grep -rn -E 'ffast-math|Ofast|funsafe-math' <你的项目>/ --include=*.txt --include=*.cmake

# 2. 跑探针（本课附带）
gcc -O2 ftz_probe.c -o probe && ./probe

# 3. 看结果
#    第 1 节 bits 列是 [SUBNORMAL] -> 正常
#    第 4 节 FTZ/DAZ/FZ = 0     -> 正常
```

### 7.7 实例：真实项目里的 fast-math（本仓库实测）

在 MNN 源码里搜到的实际用法：

| 位置 | 情况 | 风险 |
|---|---|---|
| `source/backend/cpu/riscv/CMakeLists.txt` | **`MNN_RVV_FAST_MATH` 可选开关**，开了会给 `MNNRVV` 加 `-ffast-math` | RISC-V 平台开了就 FTZ |
| `cmake/ios.toolchain.cmake` | **Release 配置硬编码 `-O3 -ffast-math`** | iOS 部署会 FTZ |
| `source/backend/cpu/CMakeLists.txt`（主） | **未开** | Android/ARM 默认安全 ✅ |

**结论**：你在 **Android/ARM 上用 MNN 是安全的**（默认 IEEE）。
但如果以后做 **iOS 或 RISC-V 部署**，或者显式打开了 `MNN_RVV_FAST_MATH`，就要留意。

> 💡 **一条通用建议**：拿到任何第三方推理库，先 grep 一遍 `fast-math`。
> 这类选项经常藏在 toolchain.cmake 或某个平台的子目录里，**不会出现在主 CMakeLists**。

### 7.8 顺手验证

```bash
gcc -O2 ftz_probe.c -o probe && ./probe                    # 基线：应全部正常
gcc -O2 -ffast-math ftz_probe.c -o p2 && ./p2              # 应看到 FTZ/DAZ=1
./p2 --ieee                                                # 应救回 subnormal
aarch64-linux-gnu-gcc -O2 -static ftz_probe.c -o pa
qemu-aarch64-static ./pa                                   # ARM 上同样验证
```

## 8. 跑一下

```bash
python3 float_bits.py subnormal        # 纯算法模型下的 subnormal 行为
gcc -O2 ftz_probe.c -o probe && ./probe   # 真实硬件/编译器的 FTZ 行为
```

想自己验证 FTZ 的影响，脚本里 `encode()` 有个开关：

```python
to_format(1e-7, 5, 10, 15, subnormal=True)    # 1.1921e-07（IEEE）
to_format(1e-7, 5, 10, 15, subnormal=False)   # 0（模拟 FTZ 硬件）
```

---

## 9. 承上启下

现在我们知道：

- 小于 $2^{-14}$ 的数会掉进 subnormal，精度从 11 位退化到个位数
- 如果编译器开了 `fast-math`（或硬件 FTZ），直接变 0（见第 7 节实测）

**下一个问题**：量化的 scale 有多大？

```python
scale = max(|w|) / 127
```

如果模型小一点、权重小一点，scale 很容易落在 $10^{-6} \sim 10^{-8}$ 这个区间——**正好是 subnormal 区甚至以下**。

这就是第 03 章：**为什么标准 fp16 装不了量化 scale，以及 `e0` 可配的私有格式怎么救。**
