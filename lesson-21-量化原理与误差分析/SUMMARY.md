# Lesson 21: 量化原理与误差分析 总结

## 课程目标

1. 量化数学：scale/zero_point 公式、对称/非对称、per-tensor/per-channel
2. 量化误差的三大规律：位宽杠杆、分布天花板、通道尺度解药
3. 定标（scaling）：浮点 scale → multiplier + shift 的硬件整数路径
4. 真实模型量化体检：用中间值分布预判量化敏感层

**呼应 Lesson 20**：量化后的 int8 输出为什么需要 int32 累加（多级累加）、`sadalp` 在做什么——上下游打通。

---

## 一、量化核心公式

```
对称量化 (zero_point=0):  q = clamp(round(x / scale))        scale = max(|x|) / 127
非对称量化:              q = clamp(round(x / scale) + zp)    scale = (max-min) / 255
反量化:                  x_hat = (q - zero_point) * scale
```

### 分母 127 / 255 的来历

分母 = 目标整数类型的阶梯数：

| 组合 | 整数范围 | 分母 | 说明 |
|---|---|---|---|
| 对称 + int8 | [-128, 127] | 127 | `2^7-1`，只映射幅度 |
| 非对称 + uint8 | [0, 255] | 255 | `2^8-1`，映射整个跨度 |

**两对独立概念**（常见误区）：
- **对称/非对称**：零点怎么放。对称 0→0 固定；非对称 0→可平移（zero_point）
- **int8/uint8**：结果装哪个盒子。int8 [-128,127] 有正负；uint8 [0,255] 非负

任意组合合法。工程惯例：**权重用对称+int8**（省 zero_point），**ReLU 后激活用非对称+uint8**（不浪费负半区间）；ONNX QDQ 常用 int8 非对称（分母 254）。

### 量化步骤（以 [0.1, 0.9] → uint8 为例）

```
scale       = (0.9-0.1)/255 = 0.0031373   每个阶梯代表多大
zero_point  = round(0 - 0.1/0.0031373) = -32   零点平移, 让 min → 0
q(0.1) = round(0.1/0.0031373) + (-32) = 0      最小值 → 0
q(0.9) = round(0.9/0.0031373) + (-32) = 255    最大值 → 255
x_hat = (q - (-32)) * 0.0031373                反量化
```

设计目标：**min → 0，max → 255，中间线性铺满**——这是非对称量化"把窄范围铺满整个整数域、不浪费阶梯"的意义。

---

## 二、量化误差的三大规律（通用知识）

### 规律 1：位宽是指数杠杆（6dB/bit）

每增加 1 bit，量化 SNR 提升约 6dB。推导：

- 量化舍入误差均匀分布在 ±scale/2，标准差 ≈ scale/√12
- 每 +1 bit，scale 减半 → 误差能量变为 1/4 → SNR 提升 10·log10(4) ≈ 6.02 dB

通用公式（含分布项）：

$$\text{SNR} \approx 6.02 \times \text{bits} + 10.8 - 20\log_{10}\left(\frac{x_{peak}}{x_{rms}}\right)$$

- `6.02×bits`：位宽贡献（每 bit 6dB）
- 第三项：数据分布贡献（见规律 2）

**结论**：加 bit 是"指数级"改善，改分布是"常数级"改善。

### 规律 2：分布是常数天花板（长尾为何差）

量化精度受"峰值与 RMS 之比"（crest factor）限制：

$$\text{SNR 损失} = 20\log_{10}\left(\frac{x_{peak}}{x_{rms}}\right)$$

- 均匀分布：peak/rms ≈ √3 ≈ 1.73 → 损失 ~4.8dB（接近理论最优）
- 长尾分布：peak/rms 大（5~10）→ 损失 14~20dB

**根因**：scale 由峰值决定（`scale = x_peak/127`），但能量由大量小值贡献。峰值大 → scale 大 → 小值量化粒度粗 → 大量小值挤进少数阶梯（甚至全变 0）→ 相对误差大。

**两类误差**：
- **clip error**：超出 [-128,127] 被截断（由峰值决定，长尾严重）
- **rounding error**：步长内舍入（由 scale 决定，约 0.29×scale）

**结论**：量化"怕峰值大、值域宽"——这是激活归一化（BN）的动机之一。

### 规律 3：per-channel 是通道尺度差异的解药

per-channel 相对 per-tensor 的增益，取决于"各通道尺度差异有多大"：

- 尺度一致（归一化后权重）：per-channel ≈ per-tensor，无增益
- 尺度差异大（depthwise conv、BN 折叠后）：per-channel 显著更好

**根因**：per-tensor 的 scale 被最大通道主导，小通道只分到几个阶梯；per-channel 每通道独立 scale，各自用满 256 级。

**使用判断**：通道间动态范围差异 >10x 时 per-channel 收益明显；差异小时没必要（per-channel 参数多、实现贵）。

---

## 三、定标（scaling）：浮点 scale → multiplier + shift

### 为什么需要定标

硬件（NPU/DSP/MAC 阵列）不做浮点乘，把 scale 表示成 `multiplier / 2^shift`：

```
浮点版:  x_hat = q * scale                            (浮点乘, 贵)
定标版:  x_hat = ((q - zp) * mult + 2^(shift-1)) >> shift   (整数乘+移位, 便宜)
```

`+ 2^(shift-1)` 是加半做四舍五入。

### shift 的确定

由"multiplier 不溢出位宽"和"尽量用满位宽"两个约束锁定：

```
目标: scale × 2^shift ≈ 2^(bits-1)     # multiplier 停在位宽一半, 留防溢出余量
解出: shift = bits - ceil(log2(scale)) - 1
```

- `-1`：防溢出余量
- `ceil`：保证整数且偏大（multiplier 偏大 → 精度高）
- 经验默认 `shift = bits` 简单够用；严格最优按公式

### multiplier 位宽决定逼近精度

| multiplier 位宽 | 范围 | 逼近精度 |
|---|---|---|
| 16 bit | 0~65535 | 够用（与浮点无差） |
| 32 bit | 0~43亿 | 比 float32 还准 |

**核心结论：定标不粗糙**。量化舍入误差（主导）远大于定标逼近误差（次要）——16bit multiplier 的定标 SNR 与浮点 scale 完全一致。粗糙与否取决于 multiplier 位宽，不是移位本身。

### 依赖链：离线/在线分离

```
原始浮点真值 x → 统计(min/max) → scale → 定标转换 → (multiplier, shift)
```

- scale 一旦算出，真值退场；multiplier/shift 一旦算出，连 scale 都退场
- **推理时只有整数 (q, zp, mult, shift)** —— 这就是"浮点模型 → 纯整数模型"
- 对应实际：转换工具（QNN）离线算好每通道 scale/shift_count，推理端只使用

### 公式法 vs 扫描法

| | 公式法 `bits-ceil(log2(scale))-1` | 扫描法 `scale_to_fixed` |
|---|---|---|
| 目标 | multiplier 接近位宽上限 | 逼近误差最小 |
| 复杂度 | O(1) | O(bits) 扫 2·bits 个 shift |
| 适用 | 理解/实验 | 产品/框架（TFLite QuantizeMultiplier） |

扫描法逐 shift 验证并处理 min/max 位宽约束；公式法 O(1) 快、够好，8-bit 下两者结果几乎无差。

---

## 四、量化指标选型

| 指标 | 衡量什么 | 适用阶段 |
|---|---|---|
| SNR（相对误差） | 每层量化损多少 | 定位问题层 |
| MSE（绝对误差） | 平均平方误差 | 需要绝对误差（不跨层比） |
| KL 散度 | 两个分布差异 | 校准 scale（PTQ 的 calibration） |
| 任务指标（WER/ACC） | 模型实际效果 | 端到端验证 |

**三个阶段的三种指标**：定位问题（SNR）→ 定 scale（KL）→ 验结果（任务指标）。

**SNR 怎么算**：`SNR(dB) = 20·log10(||ref||₂ / ||err||₂)`。`||·||₂` 是 L2 范数
（能量开根号），配合系数 20 正好还原能量比定义 `10·log10(信号能量/误差能量)`；
数值越大越精确（每 6dB = 误差能量减半，每 20dB = 减到 1/10）。

---

## 五、真实模型体检方法（PTQ 预判）

**方法**：dump 模型 fp32 中间值 → 每层算分布特征（near0 + kurtosis）→ 模拟 per-tensor/per-channel 量化 → 对比 SNR。

**分布特征**：
- `near0`：多大比例元素挤在 0 附近（相对 max 的 1%）→ 长尾/稀疏度
- `kurtosis`：分布尾巴多重（4 次矩 / 方差²）→ 正态=3，长尾>>3

**敏感层判据**：
- snr_pt < 25dB → 量化会崩
- near0 > 80% 或 kurt 极大 → 长尾分布 → 需 per-channel / 非对称 / 保 fp32

**应用**：模型转 QNN 后若精度崩，先查量化敏感层（Add/Conv/Slice 等通道尺度差异大的层）的量化参数——对应转换工具给的每通道 scale/shift_count。

---

## 六、真实权重定标转换（实验 3：fsmn 的 conv_left / conv_right）

**目标**：从真实模型权重出发，重走一遍"QNN 给 scale/shift_count"的完整链路，验证定标无损。

### 数据来源

用 onnx_layer_dump 工具 dump fsmn 的 conv 权重（`fsmn_out/weights/*.bin` + `*.meta.json`）：

| 权重 | shape | 范围 |
|---|---|---|
| conv_left | (128, 1, 10, 1) | [-0.8877, 0.5221] |
| conv_right | (128, 1, 2, 1) | [-1.1308, 0.8959] |

C_out=128 → **128 组量化参数**（每输出通道一组）。

### 完整链路（每通道）

```
通道权重 → max(|x|) → per-channel scale → 扫描法定标 → (multiplier, shift)
```

1. **per-channel scale**：每输出通道独立 `max(|x|)/127`。conv_left 128 个 scale 在 [0.0005, 0.0070]；conv_right 在 [0.0007, 0.0089]。**通道间差 ~14 倍** → 正是"per-tensor 被最大通道主导、小通道浪费阶梯"的实锤。
2. **定标**：scale → multiplier/shift。conv_right 实测 multiplier ∈ [2019, 64563]，shift ∈ [19, 26]，全部落在 16bit 位宽内（max 64563 < 65535）。
3. **验证定标无损**：对比两条反量化路径——
   - 浮点版：`x_hat = q * scale`
   - 定标版：`x_hat = q * multiplier / 2^shift`（用浮点除法，不是 `>>` 取整）
   - 结果：conv_left 3.434e-03 vs 3.435e-03；conv_right 3.575e-03 vs 3.574e-03——
**同量级，误差 < 1 个 quant 阶梯**。

---

## 七、整数推理闭环（实验 4：真实激活 + 权重 → 全程整数）

**目标**：用 dump 的真实中间 tensor + 真实权重，把"量化 → 整数卷积 → requantize"
完整走一遍，验证部署到 NPU/DSP 的样子。

### 链路（必须查 layers/*.json，不能靠 tensor 名字猜！）

```
Concat(cache拼接) → Slice_1 [1,128,13,1] → conv_left [1,128,4,1]
```

- conv_left 是 **depthwise conv**（weight (128,1,10,1)，group=128），无 padding
  （`13 - 10 + 1 = 4`）
- 把输入当成 Transpose（4 帧）→ RMSE=2.15 全崩；Transpose 是喂 conv_right 的。
  **教训：dump 的 tensor 前缀≠图结构，谁喂谁必须查 `layers/layer_XXXX_*.json` 的
  inputs/attrs**。

### 数学主线

```
浮点卷积   y_f   = Σ(w_f·x_f)
量化后     y_int32 = Σ(w_q·x_q)                  (必须 int32 累加)
反量化     y_f   = y_int32 · scale_w · scale_x   (恒等式, 非近似!)
requantize y_q   = round(y_f / scale_y)          (输出还得是 int8, 给下一层)
合并一步   y_q   = round(y_int32 · scale_w·scale_x / scale_y)
                        └──────────┬──────────┘
                    合并成一个 multiplier+shift — QNN requantize 参数来源
```

**两个 scale 的分工**：`scale_w·scale_x` 负责"还原回浮点"（反量化），`scale_y` 负责
"再量化成 int8"（输出量化）。硬件把这两个动作合成一个整数乘加移位。

### 实测数据（SNR vs 浮点真值）

| 方法 | SNR(dB) | 结论 |
|---|---|---|
| A 理论 scale 传播（浮点乘） | 35.83 | 纯量化误差上界 |
| B 定标逼近 `mult/2^shift`（浮点除法） | 35.83 | **16bit 定标逼近无损**（实测逼近误差仅 1.35e-5，是量化误差的 1/1000） |
| B2 错误示范：浮点域 `>>` 取整 | 17.62 | **坑**：取整误差 0.5 相对幅度(~10)太大 |
| D 完整 requantize → int8（还原浮点） | 36.37 | 除以 scale_y 也只损失量化误差 |

- A vs B 的最大相对差 1.35e-5 = 定标逼近误差；量化误差 ~1e-2（SNR 35.83dB）。
  **逼近误差比量化误差小三个数量级 → "16bit 定标无损"成立**
- D 比 A 高 0.5dB 不是 D 更准：`scale_y` 由 y_ref 本身统计（上帝视角），D 的还原
  值贴着真值走；0.5dB 在量化误差波动范围内，无统计意义
- 实际峰值 ±16409 vs 理论边界 ±161290（K·127²）——**硬件按最坏情况设计 int32，
  不能赌输入**（本次数据 int16 恰好没溢出，但结论不变）
- "int8 输出一致率 29.69%、int8 域 SNR 26dB"衡量的是 y_q（整数路径）vs y_q_ref
  （直接量化真值）的一致度，**不是** vs 浮点真值；vs 浮点真值看 35~37dB。
  边界翻转是量化误差的正常现象，单点一致率无意义，看 SNR（呼应"指标选型"）

### 为什么取整必须在 int8 域做（B2 的教训）

- 浮点小数值域（±10）做 `>>` 取整：0.5 的取整误差 = 5% 幅度 → SNR 掉 18dB
- 硬件做法：先除以 `scale_y` 放大到 int8 域（±127）再取整：0.5 相对 127 < 0.4%
- **定标/取整的"粒度"由量化位宽决定，必须在量化域做**——这就是 requantize
  合并 multiplier 的物理意义（放大 + 取整一步完成）


