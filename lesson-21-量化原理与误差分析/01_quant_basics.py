# Lesson 21 实验 1: 量化数学 + 误差模拟（纯 numpy，无框架依赖）
#
# 目标:
#   1. 用 numpy 从零实现 int8 量化（对称/非对称、per-tensor/per-channel）
#   2. 量化→反量化往返，测量误差，理解误差来源
#   3. 观察激活分布对量化误差的影响（均匀 vs 长尾）
#
# 运行: python 01_quant_basics.py


# ----------- 1. 量化核心公式 ----------- #
# 对称量化 (zero_point=0):  q     = clamp(round(x / scale))
# 非对称量化:               q     = clamp(round(x / scale) + zero_point)
# 反量化:                   x_hat = (q - zero_point) * scale
# scale = (max-min)/255(非对称) 或max(|x|) / 127(对称)

# ⚠️ 分母 127 / 255 的来历（核心概念）:
#   分母 = 目标整数类型的"阶梯数"，由量化结果用什么整数类型决定：
#     int8  范围 [-128, 127]  → 对称映射 max(|x|) → 127    (2^7-1)
#     uint8 范围 [0, 255]     → 非对称映射 (max-min) → 255 (2^8-1)
#   ⚠️ 注意：非对称 ≠ uint8！非对称只是"带 zero_point 平移"，
#      整数类型可以是 uint8（分母 255）也可以是 int8（分母 254，ONNX QDQ 风格）。
#      本文件演示：对称 → int8（127），非对称 → uint8（255）。

import numpy as np


def quantize_symmetric(x, bits=8):
    """对称量化: 整数范围 [-2^(bits-1), 2^(bits-1)-1]
    bits=8 → int8 [-128, 127] → scale = max(|x|) / 127
    bits=16 → int16 → scale = max(|x|) / 32767
    """
    q_max = 2 ** (bits - 1) - 1  # 127 / 32767
    q_min = -(2 ** (bits - 1))  # -128 / -32768
    # 把 x 映射到 [-127, 127]，然后四舍五入取整，最后 clamp 到 [-128, 127]
    scale = np.max(np.abs(x)) / q_max
    q = np.clip(np.round(x / scale), q_min, q_max).astype(np.int16)
    return q, scale


def quantize_asymmetric(x, bits=8):
    """非对称量化: 整数范围 [0, 2^bits-1]
    bits=8 → uint8 [0, 255] → scale = (max-min) / 255
    """
    q_min, q_max = 0, 2**bits - 1  # 0, 255
    x_min, x_max = np.min(x), np.max(x)
    scale = (x_max - x_min) / (q_max - q_min)  # (max-min)/255
    zero_point = np.round(q_min - x_min / scale).astype(np.int32)
    q = np.clip(np.round(x / scale) + zero_point, q_min, q_max).astype(np.uint8)
    return q, scale, zero_point


def dequantize(q, scale, zero_point=0):
    """反量化: x_hat = (q - zero_point) * scale"""
    return (q.astype(np.float32) - zero_point) * scale


def measure(x, x_hat):
    """量化误差指标"""
    err = x - x_hat
    return {
        "max_abs": np.max(np.abs(err)),
        "rmse": np.sqrt(np.mean(err**2)),
        "snr_db": 20 * np.log10(np.linalg.norm(x) / (np.linalg.norm(err) + 1e-12)),
    }


def float_to_fixed(scale, mult_bits):
    """把浮点 scale 转成 (multiplier, shift): scale ≈ multiplier / 2^shift
    定标（scaling）核心概念：
      - 硬件不想做浮点乘，把 scale 表示成「整数 multiplier / 2^shift」
      - 反量化从 x*q*scale 变成 (x*q*multiplier) >> shift（整数乘+移位）
      - multiplier 位宽越多，逼近越准（16bit ≈ 0~65535，32bit ≈ 0~43亿）
    """
    shift = mult_bits  # shift 取足够大，让 multiplier 用满位数
    multiplier = round(scale * (2**shift))
    scale_fixed = multiplier / (2**shift)
    return multiplier, shift, scale_fixed


# ---------- 2. 误差来源实验 ----------
print("=" * 60)
print("实验 A: 分布形状对量化误差的影响")
print("=" * 60)
rng = np.random.default_rng(42)
n = 10000

# 均匀分布
x_uniform = rng.uniform(-1, 1, size=n).astype(np.float32)
# 长尾分布，大部分在0附近，少量大值
x_tailed = np.concatenate(
    [
        rng.normal(0, 0.05, int(n * 0.95)),  # 95% 小值
        rng.uniform(0.5, 1.0, int(n * 0.05)),  # 5% 大值
    ]
).astype(np.float32)


for name, x in [("均匀分布", x_uniform), ("长尾分布", x_tailed)]:
    q_sym, s_sym = quantize_symmetric(x)
    x_hat_sym = dequantize(q_sym, s_sym)
    q_asym, s_asym, zp_asym = quantize_asymmetric(x)
    x_hat_asym = dequantize(q_asym, s_asym, zp_asym)

    m_sym = measure(x, x_hat_sym)
    m_asym = measure(x, x_hat_asym)
    print(f"\n[{name}]")
    print(
        f"  对称量化:   max_abs={m_sym['max_abs']:.6f}  rmse={m_sym['rmse']:.6f}  snr={m_sym['snr_db']:.1f} dB"
    )
    print(
        f"  非对称量化: max_abs={m_asym['max_abs']:.6f}  rmse={m_asym['rmse']:.6f}  snr={m_asym['snr_db']:.1f} dB"
    )
    print(f"  scale: 对称={s_sym:.6f}  非对称={s_asym:.6f}")


# ---------- 3. 量化误差 vs 位宽 ----------
print("\n" + "=" * 60)
print("实验 B: 位宽对误差的影响 (4/8/16 bit)")
print("=" * 60)

for bits in [4, 8, 16]:
    q, s = quantize_symmetric(x_uniform, bits=bits)
    x_hat = dequantize(q, s)
    m = measure(x_uniform, x_hat)
    print(f"  {bits:2d}-bit: max_abs={m['max_abs']:.6f}  snr={m['snr_db']:.1f} dB")
# 预期: 每 +1 bit, snr +~6dB（6dB/bit 法则）


# ---------- 4. per-tensor vs per-channel ----------
print("\n" + "=" * 60)
print("实验 C: per-tensor vs per-channel（权重矩阵量化）")
print("=" * 60)

# 权重: [C_out, C_in] = [64, 256]
W = rng.normal(0, 1, (64, 256)).astype(np.float32)
# 故意让不同 channel 的尺度差 100 倍（模拟真实权重分布差异）
W = W * np.expand_dims(np.linspace(0.01, 1.0, 64), axis=1)

# per-tensor: 整个矩阵共用一个 scale
q_pt, s_pt = quantize_symmetric(W)
W_hat_pt = dequantize(q_pt, s_pt)
m_pt = measure(W, W_hat_pt)

# per-channel: 每个输出 channel 一个 scale
scales = np.max(np.abs(W), axis=1) / 127.0  # [C_out]
q_pc = np.clip(np.round(W / scales[:, None]), -128, 127).astype(np.int8)
W_hat_pc = q_pc.astype(np.float32) * scales[:, None]
m_pc = measure(W, W_hat_pc)

print(f"  per-tensor:  snr={m_pt['snr_db']:.1f} dB")
print(f"  per-channel: snr={m_pc['snr_db']:.1f} dB")


# ---------- 5. 浮点 scale vs 定标 (multiplier+shift) ----------
print("\n" + "=" * 60)
print("实验 D: 浮点 scale vs 定标 (multiplier+shift)")
print("=" * 60)
# 回答: "定标是不是不如浮点 scale？移位肯定粗糙？"
# 定标 = 把浮点 scale 表示成「整数 multiplier / 2^shift」
#   - 浮点: x_hat = q * scale          (1 次浮点乘, 贵)
#   - 定标: x_hat = (q * mult) >> shift (1 次整数乘 + 1 次移位, 便宜)
# 精度关键: multiplier 用多少位宽
#   - 16bit multiplier (0~65535): 逼近误差较小
#   - 32bit multiplier (0~43亿):  逼近误差比 float32 还小
# 结论预判: 量化舍入误差(主导) >> 定标逼近误差(次要),
#           定标 16bit 就和浮点几乎无差别, 32bit 完全无感

# 用均匀分布数据测（baseline 是浮点 scale）
q_fp, s_fp = quantize_symmetric(x_uniform)
x_hat_fp = dequantize(q_fp, s_fp)
m_fp = measure(x_uniform, x_hat_fp)

# 定标: 16-bit multiplier
mult16, shift16, s_fixed16 = float_to_fixed(s_fp, 16)
q16 = np.clip(np.round(x_uniform / s_fixed16), -128, 127).astype(np.int16)
x_hat16 = q16.astype(np.float32) * s_fixed16
m16 = measure(x_uniform, x_hat16)

# 定标: 32-bit multiplier
mult32, shift32, s_fixed32 = float_to_fixed(s_fp, 32)
q32 = np.clip(np.round(x_uniform / s_fixed32), -128, 127).astype(np.int16)
x_hat32 = q32.astype(np.float32) * s_fixed32
m32 = measure(x_uniform, x_hat32)

print(
    f"  浮点 scale:  scale={s_fp:.8f}                       snr={m_fp['snr_db']:.2f} dB"
)
print(
    f"  定标16bit:   mult={mult16:6d} shift={shift16:2d}   scale≈{s_fixed16:.8f}  snr={m16['snr_db']:.2f} dB"
)
print(
    f"  定标32bit:   mult={mult32:10d} shift={shift32:2d} scale≈{s_fixed32:.8f}  snr={m32['snr_db']:.2f} dB"
)
print(f"\n  关键对比:")
print(f"    浮点 vs 定标16bit: 差 {m_fp['snr_db'] - m16['snr_db']:.2f} dB")
print(f"    浮点 vs 定标32bit: 差 {m_fp['snr_db'] - m32['snr_db']:.2f} dB")
print(f"""
  结论: 定标不粗糙!
    1. 量化舍入误差(主导, ~48dB 水平) >> 定标逼近误差(次要, 小数点后几位)
    2. 16bit multiplier 的定标 SNR 和浮点几乎一样(差 <0.01dB)
    3. 32bit multiplier 逼近误差比 float32 还小, 完全无感
    4. 硬件用定标不是"妥协", 是"整数乘+移位比浮点乘快省电, 精度还够"
    (粗糙与否取决于 multiplier 位宽, 不是移位本身)
""")


# ---------- 6. 结论 ----------
print("""
═══════════════════════════════════════════════════
实验 1 结论
═══════════════════════════════════════════════════
1. 量化误差两大来源:
   - clip error: 超出 [-128,127] 范围被截断 (由 max|x| 决定, 长尾分布严重)
   - rounding error: 量化步长内的舍入 (由 scale 决定, 均匀分布约 0.29*scale)
2. 长尾分布 (激活常见) 比均匀分布差:
   scale 被少数大值拉大 → 大量小值量化成 0 → 小值信息全丢
3. 6dB/bit 法则: 位宽每 +1, SNR 约 +6dB
4. per-channel 解决"各通道尺度差异大"问题:
   每通道独立 scale → 小尺度通道不被大尺度通道拖累
5. 定标(multiplier+shift) ≈ 浮点 scale, 不粗糙:
   量化舍入误差是主导, 定标逼近是次要; 16bit multiplier 就够用
6. 这就是为什么:
   - ReLU 后的激活用非对称 (值域 [0,+x]) 更合适
   - 权重常用 per-channel 对称 (per-tensor 会丢小通道精度)
   - 量化敏感的层 (如 depthwise conv 的通道) 要格外小心
""")
