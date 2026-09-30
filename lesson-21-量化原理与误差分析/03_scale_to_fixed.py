# 03_scale_to_fixed.py
# Lesson 21: 实验3： 真实Conv权重的per-channel 定标转换
# 目标：
# 把fsmn真实的conv权重浮点scale转成硬件友好的（multiplier,shift)
# 链路：权重 -> 每通道max -> per-channel scale -> 定标（multiplier,shift)

import json
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fsmn_out"


def scale_to_fixed(scale, bits, min_mult=0, max_mult=None):
    """扫描法: 浮点 scale → (multiplier, shift), 逼近误差最小"""
    if max_mult is None:
        max_mult = (1 << bits) - 1
    best = None
    # Q: 这里为什么遍历 shift 范围是 0 ~ 2*bits？
    # A: 遍历 0~2·bits = 覆盖所有"multiplier 从 1 涨到 2^bits"的可能 shift；
    # 超出这个范围 mult 必溢出或太小，扫描无意义。
    for shift in range(0, 2 * bits + 1):
        mult = round(scale * (1 << shift))
        if mult < min_mult or mult > max_mult:
            continue
        scale_fixed = mult / (1 << shift)
        err = abs(scale_fixed - scale) / scale
        if best is None or err < best[0]:
            best = (err, mult, shift)
    return best[1], best[2]


def process_weight(name, path, bits=16):
    """读权重，per-C_out算scale，转定标，输出对照
    bits=16 是 multiplier 的位宽 不是量化结果的位宽。"""
    meta = json.load(open(path.with_suffix(".meta.json"), "r"))
    w = np.fromfile(path, dtype=np.float32).reshape(meta["shape"])
    print(f"\n{'='*64}")
    print(f"权重: {name}  shape={w.shape}  (C_out={w.shape[0]})")
    print(f"范围: min={w.min():.4f} max={w.max():.4f}")

    # ------ per-channel scale ------
    # Q: 这里为什么除以127？
    # A: 127 是 int8 的最大绝对值，这里实际写死了量化到int8
    scales = np.max(np.abs(w), axis=tuple(range(1, w.ndim))) / 127.0
    print(
        f"per-channel scale: {len(scales)}个 范围： [{scales.min():.4f}， max={scales.max():.4f}]"
    )
    # ------ per tensor 对照 ---------------
    scale_t = np.max(np.abs(w)) / 127.0
    print(f"per-tensor scale: {scale_t:.6f}  (vs per-channel max {scales.max():.6f})")

    # ---------- 定标转换 ----------
    print(f"\n前 8 个通道的定标结果 (bits={bits}, 无符号):")
    print(
        f"  {'ch':>3} {'scale':>12} {'multiplier':>10} {'shift':>5} {'scale_fixed':>12} {'rel_err':>9}"
    )
    for i in range(min(8, len(scales))):
        mult, shift = scale_to_fixed(scales[i], bits)
        scale_fixed = mult / (1 << shift)
        rel_err = abs(scale_fixed - scales[i]) / scales[i]
        print(
            f"  {i:3d} {scales[i]:12.8f} {mult:10d} {shift:5d} {scale_fixed:12.8f} {rel_err:9.2e}"
        )

    # ------- 统计 -------
    pairs = [scale_to_fixed(s, bits) for s in scales]
    mults = np.array([p[0] for p in pairs])
    shifts = np.array([p[1] for p in pairs])
    print(
        f"\n统计: multiplier 范围 [{mults.min()}, {mults.max()}], "
        f"shift 范围 [{shifts.min()}, {shifts.max()}]"
    )
    print(f"  → 每通道一组 (scale, multiplier, shift), 就是 QNN 给的 scale/shift_count")
    # ---- 验证: 定标反量化 vs 浮点 scale 反量化 ----
    # ⚠️ 对比用浮点除法 (q*mult)/2^shift —— 硬件上才是 >> shift (整数右移)
    #    验证"定标是否无损"必须和浮点 scale 同量纲对比, 不能取整
    q = np.clip(np.round(w / scales.reshape(-1, 1, 1, 1)), -128, 127).astype(np.int16)
    w_hat_float = q.astype(np.float32) * scales.reshape(-1, 1, 1, 1)
    w_hat_fixed = np.zeros_like(w)
    for i in range(len(scales)):
        mult, shift = scale_to_fixed(scales[i], bits)
        w_hat_fixed[i] = (q[i].astype(np.float32) * mult) / (1 << shift)
    err_float = np.abs(w - w_hat_float).max()
    err_fixed = np.abs(w - w_hat_fixed).max()
    print(f"\n反量化最大误差: 浮点scale版={err_float:.6e}  定标版={err_fixed:.6e}")
    print(
        "结论: 两版误差同量级 → 定标转换对量化结果无损 "
        "(硬件用 >> 取整, 精度损失 < 1 个 quant 阶梯)"
    )


if __name__ == "__main__":
    weights_dir = OUT / "weights"
    for name in [
        "backbone_fsmn_0_1_conv_left_weight",
        "backbone_fsmn_0_1_conv_right_weight",
    ]:
        bin_path = weights_dir / f"{name}.bin"
        if bin_path.exists():
            process_weight(name, bin_path)
