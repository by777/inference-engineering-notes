# Lesson 20: MNN 手写 NEON vs TVM 生成 总结

## 课程目标

1. **看懂 ARM64 NEON 反汇编**——从 `objdump -d` 的机器码读出"编译器在干什么"
2. **对比两条算子实现路径**——MNN 手写 `.S` 汇编 vs TVM+LLVM 自动生成
3. **找到"编译器边界"**——f32 编译器够强，int8 必须手写，为什么？
4. **理解调度（循环顺序）对 SIMD 的生死影响**——谁放内层，决定 lane 用不用得上
5. **澄清 AutoTVM/XGBoost 的角色**——它解决"选配置"，不解决"发明布局"

---

## 一、两个实验回顾

| 实验 | 文件 | 内容 | 关键产出 |
|------|------|------|----------|
| 01 | `01_tvm_generate_arm_neon.py` | TVM 编译 f32/int8 matmul → 反汇编对比 MNN 手写 | f32 生成 30+ 条 `fmla`，int8 只生成 1 条 `smull2` |
| 02 | `02_tvm_arm_matmul_benchmark.py` + `02_main_arm.cpp` | 4 个调度编译 ARM64 → 推板子实测 | naive 0.2x / opt 2.4x / parallel 0.2x / opt_parallel 4.6x |

### 实验 1：反汇编证据——f32 vs int8 的差距

**f32 版**（`tvm_matmul_arm_neon_only.txt`）：

```asm
fmla v7.4s, v2.4s, v0.s[0]   ; A[i,k] 标量广播 × B 的 4 lane → C 向量累加
fmla v15.4s, v4.4s, v0.s[0]  ; 30+ 条，用满 v0~v31 全部寄存器
```

- 4-lane 融合乘加 + 标量广播（`v0.s[0]` = 广播的 A[i,k]）
- 32 个 NEON 寄存器全用上做 register tiling
- 唯一瑕疵：unroll 过头 → 部分累加器 `str q, [sp]` 溢出到栈

**int8 版**（`tvm_matmul_int8_neon_only.txt`）：

```asm
ld1r  {v1.8b}, [x16]        ; B 只能"广播加载 1 个元素"
sxtl  v0.8h, v0.8b          ; int8 → int16
sxtl  v1.8h, v1.8b
smull2 v0.4s, v0.8h, v1.8h  ; ⭐ 唯一的向量乘法：4 个 int32
mov   w16, v0.s[3]          ; ⭐⭐ 只取第 3 个 lane！
add   w13, w16, w13         ; ⭐⭐ 标量加法累加
```

- **整个计算核心只有 1 条向量乘法指令**
- **`smull2` 算出的 4 个 lane 只取 1 个**（`v0.s[3]`），75% 算力白费
- C 用 `ldr w13 / str w13`（4 字节标量）读写，对比 f32 的 `stp q`（32 字节）

---

## 二、核心概念速查

### NEON 基础：寄存器 / lane / 后缀

```
NEON 寄存器 = 128-bit（不是 512！此前 NPU 才是 512-bit）
后缀 = 切分粒度，lane 数 × lane 宽度 = 128 固定：
  .16b = 16 个字节（int8）      → 16 lanes
  .8h  = 8 个半字（int16）      → 8 lanes
  .4s  = 4 个单字（int32/fp32） → 4 lanes
  .2d  = 2 个双字（int64/fp64） → 2 lanes

lane = 向量寄存器里的一条"道"，一条指令同时操作所有 lane（SIMD）
```

**此前 NPU 对照**（你的硬件直觉直接迁移）：
| | 此前 NPU | ARM NEON |
|---|---|---|
| 寄存器 | 512-bit | 128-bit |
| fp32 lane | 16 | 4 |
| int8 lane | 64 | 16 |
| 手写方式 | 宏+脚本翻译 asm / intrinsics | `.S` / LLVM 生成 |

### 关键指令对照表

| 指令 | 含义 | 对应此前 NPU |
|---|---|---|
| `ld1 {v0.16b}, [x]` | 向量加载（一次 16B） | `LD x, 16B` |
| `st1 {v2.16b}, [x]` | 向量存储 | `ST x, 16B` |
| `smull v0.8h, v0.8b, v1.8b` | 8 路 int8×int8→int16 乘 | `MAC 阵列 8x8` |
| `smlal2 v0.8h, v0.16b, v1.16b` | 高 8 路乘 + 累加 | MAC 阵列累加 |
| `sadalp v16.4s, v8.8h` | pairwise 累加 int16→int32 | 累加器 ADD |
| `fmla v0.4s, v1.4s, v0.s[0]` | 4-lane 融合乘加 + 标量广播 | MAC 阵列+ 广播 |
| `dup v31.16b, w6` | 标量→向量广播 | broadcast |
| `sxtl` | 符号扩展（int8→int16→int32） | 位宽转换 |

---

## 三、实验 2：板上实测 4 个调度（某 arm64 开发板 = 某 arm64 移动平台, 2026-08-20）

**实测结果（同进程 naive 基准，3 轮取 min）**：

| 版本 | best (ms) | GFLOPS | vs naive |
|------|-----------|--------|----------|
| naive (i,j,k) | 11.80 | 2.9 | 1.0x |
| opt (i,k,j 向量化) | 1.30 | 25.8 | **9.1x** |
| parallel (i 并行，k 最内) | 15.58 | 2.2 | **0.8x（比 naive 慢）** |
| **opt_parallel** | **0.52** | **65.0** | **23.2x** |

**结论（与 Lesson 17 x86 完全一致，ARM 板上复现）**：
- opt 仅 j 向量化 → 9.1x；opt_parallel 三刀全上 → 23.2x（x86 为 37.7x，ARM 单核算力/SIMD 宽度更弱，趋势一致）
- **parallel 比 naive 还慢**——并行不解决 cache 问题，反而放大（16 核抢带宽跑 k 最内层低效循环）
- "重排先行、并行后上"顺序有效：opt 1.30 → opt_parallel 0.52，加 i 并行再提速 ~2.5x

**测时方法学（踩坑记录）**：
- 板子无法锁频（`sched_setaffinity failed`），CPU 频率每个进程启动时不同
- 加速比必须**同进程内对比**：加载 naive 基准 + 被测模块，同频率状态，公平
- 所有测时 3 轮取 min（命中最高频），warmup 前置（先跑几轮拉满频率）
- 验证：naive 自比 = 1.0x、ref(C 标量) 与 naive best 对齐（11.79 vs 11.80）→ 测时正确

---

## 四、核心结论：编译器边界在哪里

### 结论 1：f32 编译器够强，int8 编译器拉胯

```
f32 matmul → LLVM 自动生成 30+ 条 fmla，寄存器分块漂亮 ✅
int8 matmul → LLVM 只生成 1 条 smull2，4 lane 用 1 个 ❌
```

**准确归因**：不是 TVM 前端调度错，是 **LLVM 后端对 int8 GEMM 缺少布局感知的算法级重构**。f32 有成熟的 autovectorizer + fmla 融合；int8 需要"16x4 分块 + 多级累加 + 布局认知"，这是算法级优化，LLVM 只会做最保守的"一条 smull2 完事"。

### 结论 2：为什么 int8 退化？—— 根因是循环顺序，不是寄存器不够

int8 版（j 外层、ki 内层）`B[ko*8+ki, j]` 在 ki 方向 **stride=N 跨行**（每步跳 128 字节）→ B 无法向量加载 → LLVM 只能 `ld1r` 广播 + 标量累加。

```
寄存器够（8 个 int8 只占 64-bit）≠ 能优化
SIMD 真正卡在：数据能否连续向量加载 + 累加器能否向量化
```

### 结论 3：手写 vs 编译器——本质收益

| 维度 | MNN 手写 NEON | TVM+LLVM 自动生成 |
|------|--------------|------------------|
| 循环/布局 | 16x4 tile + 权重预打包（layout 钉死） | 裸 i/j/k 循环，靠 LLVM 发挥 |
| B 访问 | 预打包后连续 → `ld1` 向量加载 | 跨行 → `ld1r` 广播 |
| 累加器 | 4 个向量累加器（住在寄存器） | 标量累加（每轮 reduce） |
| 量化 fusion | scale/bias/relu 内联 GEMM 尾部 | 需要 pass 或手写 |
| 性能 | 逼近硬件 95%+ | 80~95%（需 autotuning） |

**手写汇编的本质收益 = layout-tied 调度 + 量化算子 fusion。**

### 结论 4：AutoTVM / XGBoost 能解决吗？——不能（部分）

```
XGBoost 解决：从"已有候选调度配置"里选最快的（tile/unroll 参数）
不解决：    "发明 16x4 布局"这个候选本身

核心陷阱：搜索空间里没有 16x4，XGBoost 搜一万次也选不出来
         （巧妇难为无米之炊）
```

真正解决路径（按工程量递增）：
1. **TVMScript 显式写 16x4 tile**（人给布局认知，你的此前 NPU 经验）
2. **AutoTVM + 手工扩展搜索空间**（把 16x4 放进候选）
3. **MetaSchedule**（进化式搜索，理论上能撞出布局，0.18 支持有限）
4. **绕开 LLVM 直接手写 asm**（MNN 的做法）

---

## 五、调度口诀（面试可背）

```
"谁放内层"的学问 = 把连续内存轴喂给最内层向量化

最内层 = 连续轴 + 可向量化（B/C 连续 → SIMD 才能发力）
中间层 = 累加轴 k（有数据依赖，不能向量化，也不挡 j）
最外层 = 并行轴 i（行独立，给 T.parallel）

Loop Interchange（循环交换）= 这个操作的专有名词
```

**同一个数学，循环顺序不同 → SIMD 效果天壤之别**：
```
for i, k, j(向量化):   → 30+ 条 fmla，lane 全用 ✅（j 连续）
for i, j, ki(向量化):  → 1 条 smull2，4 lane 用 1 个 ❌（B 跨行）
```

---

## 六、面试素材：把此前 NPU 经验"编译器化"

> 我在 此前 NPU 上手写算子时就知道编译器会浪费 lane，现在看到 TVM 在 int8 NEON 上果然把 4 个 lane 只用了 1 个。**两个平台的"编译器退化"证据我都有。**

**核心表达**：
- 此前 NPU 的「宏+asm 全控制」路径 ↔ MNN 手写 NEON
- 此前 NPU 的「intrinsics 编译器辅助」路径 ↔ TVM autotuning
- **编译器能自动化 90%，最后 10% 的布局认知必须人来给** —— 这是 AI 编译器工程师最值钱的能力
- "知道编译器在哪强（f32）、在哪弱（int8）" = 工具链工程师的核心判断力

---

## 七、踩坑清单（跨 lesson 复用）

1. `tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon")` —— 不写 `--host`
2. `tvm.lower(func, target=)` 在 0.18 没有 target 参数 → 直接 `tvm.build`
3. `export_library` 必须传 `fcompile` 用 NDK clang++ 链接（x86 ld 不认 EM_AARCH64）
4. **ABI 一致性铁律**：NDK runtime + NDK clang++ 主程序 + NDK clang++ 链接 .so，不能混 glibc
5. `tvm.build(PrimFunc)` 用 `name="default"` 让 C 端统一调用（不走 graph_executor，避开弱符号 segfault）
6. 验证 .so 依赖：`objdump -p` 看 NEEDED 应只有 `libc++_shared.so`（不能有 `libc.so.6`）
