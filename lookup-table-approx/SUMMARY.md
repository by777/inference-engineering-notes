# 查表法近似专题（LUT Approximation）

> 第五线：性能优化 / 定点化专题 —— 从 此前 NPU 回忆的 softmax 查表优化出发，
> 用纯 Python 仿真把原理、表项来源、误差账、完整流水全部跑通。
> 无板子、无硬件依赖，`python3 stepN_xxx.py` 逐个跑即可。

---

## 为什么值得单独开一章

softmax 的 `exp` 在无 FPU / 超越函数昂贵的环境（此前 NPU、Cortex-A 裸机、MCU）
是**最贵的那类操作**。查表法（LUT）是把它降到"整数移位 + 查小表 + 整数乘加"的
标准做法，也是量化部署（第 21/22 课）与定点化的地基之一。

这条线和你已有的课程互补：

| 现有线 | 视角 | 本专题补什么 |
|---|---|---|
| ONNX Runtime / TVM / MNN | 框架与编译器视角：怎么描述、怎么调度 | **数值方法视角**：单个函数怎么在硬件上高效算出 |
| MLIR | 通用 IR 机制 | 查表/近似本质是"算子内部"的优化，最终可表达为 linalg/codegen 的细节 |
| 量化课 | scale/zero-point 怎么定 | **超越函数怎么定点化**——查表是其中的核心手段 |

---

## 内容地图（6 步，每步独立可跑）

| Step | 文件 | 回答的问题 | 关键结论 |
|---|---|---|---|
| 1 | `step1_range_reduction.py` | exp 为什么能拆成"移位 + 查 [0,1)"？ | `exp(x)=2^n·2^f`，`n` 变移位、只有 `f∈[0,1)` 需要查表 |
| 2 | `step2_table_generation.py` | **表项到底怎么来的？** | 表项是离线算的：`table[k]=round(2^(k/8)·2^Q)`，板上只是查 |
| 3 | `step3_lookup_interp.py` | 高 3bit 查表 + 低 5bit 插值，误差多大？ | 全范围 max_rel≈0.1%；比纯查表好 ~90 倍，比 256 全表小 28 倍 |
| 4 | `step4_softmax_pipeline.py` | 完整 softmax 定点流水怎么串？对拍浮点 | 输出概率偏差 ~1e-4 量级；argmax 不变；动态范围大了会下溢 |
| 5 | `step5_error_improve.py` | 误差公式验证 + 怎么进一步压误差 | 误差 ≈ 线性化项主导；最优弦减半；段数↑按 N⁻² 收敛 |
| 6 | `step6_sigmoid.py` | sigmoid 怎么查表（对称性 + 饱和） | 对称性砍半表；误差 ∝ 段宽²×曲率，比 exp 大 ~6 倍 |
| 7 | `step7_tanh.py` | tanh 自己建表 vs 复用 sigmoid | 实测几乎打平：段宽减半收益被曲率大 8 倍抵消（ε∝h²·\|f''\|） |
| 8 | `step8_reciprocal.py` | 1/x 查倒数表 + 牛顿迭代 | 查表给初值 + 迭代二次收敛，2 轮到 float 全精度 |
| 9 | `step9_linear_counterexample.py` | y=2x+1 建表反例 | 建表三步模板；线性函数查表纯属浪费 |
| 10 | `step10_rsqrt.py` | 1/√x 查表 + 牛顿迭代 | 指数减半（拆偶数）+ 平方收敛（系数 3/2），全程无除法；末尾 `trace_one(4.1)` 逐步打印完整链路 |
| 11 | `step11_ln.py` | ln x 查表（range reduction 是**加法**） | `ln x = ln m + e·ln2`；表域 [1,2) → 索引**零乘法**；**ln2 的量化误差会被 e 线性放大**，必须单独提精度 |
| 12 | `step12_nonuniform.py` | **非均匀分段**到底值不值得？ | 收益由**曲率变化比**决定；ln 只差 4 倍 → **加段数完胜非均匀**；几何最优网格撞上 8bit **编码冲突**；非均匀压低了内点误差却被**端点截断**吃掉 |
| 13 | `step13_softmax_div.py` | 补上 step4 的缺口：`p_i=e_i/S` 的**定点除法** | 把 **N 次除法**换成 **1 次倒数 + N 次乘法**（约快 29 倍）；倒数用 range reduction + 查表 + 2 轮牛顿（内部 Q30）；索引**零乘法**；**Σp 截断偏差**与舍入的"偏差 vs 方差"取舍 |

> ⚠️ **rsqrt 的牛顿迭代是平方收敛** $d\to-\tfrac32d^2$，系数 $3/2$ 比 1/x 的 $1$ **更差**。
> rsqrt 迭代的价值在**无除法**，不在收敛阶。推导见 `公式推导.md`。

**建议顺序**：1 → 2 → 3 → 4 → 5 → 6 连着跑。每个脚本结尾都有小结。

---

## 5 步的核心逻辑链（30 秒版）

```
exp(x) 太贵（无 FPU）
   │
   ├─ ① range reduction: exp(x) = 2^(x·log2e) = 2^n · 2^f   (n 整数、f∈[0,1))
   │      → n 用「移位」实现（零成本），只有 f 需要函数求值
   │
   ├─ ② 建表（离线，PC 上浮点算）: table[k] = round(2^(k/8) · 2^Q)  共 9 项
   │
   ├─ ③ 查表 + 插值（板上，纯整数）:
   │      idx  = f_byte >> 5     (高 3bit 定 8 段)
   │      frac = f_byte & 31     (低 5bit 定 32 级)
   │      out  = (t[idx]·(32-frac) + t[idx+1]·frac) >> 5   ← LERP
   │
   └─ ④ 误差预算: 线性化误差 h²/8·max|f''| ≈ 1.9e-3 主导，插值级数只到线性化地板
```

---

## 术语映射（简历/面试可用）

| 口语叫法 | 正式术语 |
|---|---|
| 3bit 查表 + 5bit 插值 | 分段线性 LUT 近似（piecewise-linear LUT）+ LERP |
| 把 exp 换成 2 的幂 | 幂分解 / 指数分解（exponent decomposition） |
| 只查 [0,1) 那段 | Range Reduction（范围压缩） |
| 表建 9 个端点值 | 端点采样表（breakpoint table），非全密表 |
| 低 5bit 插值 | 线性插值（linear interpolation） |
| 全程整数无分支 | 定点化 + 时间确定性（deterministic timing） |
| 表小到能放寄存器 | 用 ALU（乘加）换存储/带宽（compute-memory tradeoff） |

---

## Roadmap（本专题第一版只做 exp/softmax，后续可延伸）

- [x] **1：exp/softmax 查表**：step1~step5 全部跑通（range reduction → 建表 → 插值 → 完整流水 → 误差数学）
- [x] **1.5：激活函数查表适用性**：见 `激活函数查表适用性.md`（判断清单 + 逐函数结论，全家桶的前置导航）
- [x] **2：超越函数全家桶**：sigmoid/tanh（softmax 的变体）、`ln`、`1/x`、`rsqrt`——
      各自的 range reduction 不同（倒数查表、对数查表配合），统一方法学
      （✅ sigmoid：`step6_sigmoid.py`；✅ tanh：`step7_tanh.py` 对比；
       ✅ 1/x：`step8_reciprocal.py`；✅ rsqrt：`step10_rsqrt.py` 查表+牛顿；
       ✅ ln：`step11_ln.py` 加法型 range reduction + 常数精度约束）

- [x] **定点除法**：softmax 分母的 `1/S` 也查倒数表 + 牛顿迭代
      （✅ `step13_softmax_div.py`：range reduction（bit_length 拆整数 S）+
      倒查表 + 2 轮牛顿（内部 Q30）→ p_i 用乘法。**至此 softmax 全程无除法**。
      附带发现：截断让 Σp 系统性 < 1，舍入能修偏差但重分配单点误差）
- [ ] **block floating point**：logits 动态范围大时按块统一指数，避免逐项下溢
- [x] **非均匀分段**：曲率大的区间段更密，同表项数误差更低
      （✅ `step12_nonuniform.py`：结论是【对 ln 不划算】—— 曲率变化比只有 4，
      非均匀内点好 2.27 倍但被端点截断吃掉，且要付 log2 索引；**加段数完胜**。
      真正值得的场景是 sigmoid/tanh 这类【饱和型】函数，曲率变化比上万）

---

## 三类 range reduction 汇总（全家桶收口）

跑完 step1~step11 后，可以把所有函数的 range reduction 归成三类：

| 类型 | 形式 | 例子 | 关键动作 |
|---|---|---|---|
| **乘法型** | $f(x)=f(m)\cdot 2^{n}$ | exp、sigmoid、tanh | $n$ 变**移位** |
| **加法型** | $f(x)=f(m)+e\cdot\ln 2$ | **ln** | $e\cdot\ln 2$ 变**乘法**（常数精度是瓶颈）|
| **幂型** | $f(x)=f(m)\cdot 2^{-e/2}$ | 1/√x（rsqrt） | 指数**减半**，须先偶化 $e$ |

**互补规律**：exp 用移位把指数"乘出去"，ln 用乘法把指数"加出去"——两者严格互逆，
所以 `step11` 的闭环（$\exp(\ln x)\approx x$）能反过来验证 `step1~5` 的 exp 流水。

**另一个分类维度**（要不要迭代）：

| 类型 | 做法 | 例子 |
|---|---|---|
| 直接够用 | 查表 + 插值 | exp、sigmoid、tanh、**ln** |
| 必须迭代 | 查表给初值 + 牛顿 | 1/x、rsqrt |

判据是**函数本身好不好用多项式/直线凑**：初值够近就不必迭代；像 $1/x$、$1/\sqrt m$
这种"曲率在整段都不小"的，靠插值精度上不去，必须加牛顿。

---

## 运行环境

- 仅 Python 3 标准库（math），无第三方依赖。
- 工作区已有 venv：`lesson-05-ORT-多输入多输出/.venv/bin/python`，直接用即可；
  或任意 `python3`。
