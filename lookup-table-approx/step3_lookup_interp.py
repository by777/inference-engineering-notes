#!/usr/bin/env python3
"""
step3: 查表 + 线性插值 —— 高 3bit 定段、低 5bit 插值，误差到底多大？

板上运行时全流程（对 f∈[0,1)，f 用 8bit 定点表示：f_byte = f·256 ∈ [0,255]）：
    idx  = f_byte >> 5      # 高 3bit → 段号 0..7
    frac = f_byte & 31      # 低 5bit → 段内 32 级位置
    out  = (t[idx]·(32-frac) + t[idx+1]·frac) >> 5    # LERP，保持 Q12

本 step 扫描全部 256 个 f_byte，对拍真值 2^(f_byte/256)：
  ① 整体精度：max_abs / max_rel 误差
  ② 误差分布：是不是"段中点最大"（线性化误差特征）
  ③ 对照实验：不做插值(纯查表) / 做插值 / 256项全表 —— 三种代价
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lut_common import FRAC_BITS, Q, gen_exp2_table, exp2_lookup_lerp


def main():
    print("=" * 72)
    print("step3  查表 + 5bit 插值：全范围误差扫描")
    print("=" * 72)

    table = gen_exp2_table()   # 9 项 Q12 表

    # ---------- 1. 一个具体例子走一遍 ----------
    print("\n[1] 具体例子: f_byte = 141（即 f = 141/256 ≈ 0.5508）")
    fb = 141
    idx = fb >> FRAC_BITS
    frac = fb & ((1 << FRAC_BITS) - 1)
    print(f"    idx  = {fb} >> 5  = {idx}   → 第 {idx} 段，端点 x∈[{idx/8:.3f},{idx/8+0.125:.3f})")
    print(f"    frac = {fb} & 31 = {frac}   → 段内 {frac}/32 处")
    lo, hi = table[idx], table[idx + 1]
    print(f"    表项 t[{idx}]={lo} (Q12), t[{idx+1}]={hi} (Q12)")
    approx = exp2_lookup_lerp(fb, table)
    truth = (2.0 ** (fb / 256.0)) * (1 << Q)   # 同 Q12 域的真值
    rel = (approx - truth) / truth
    print(f"    插值结果 = {approx} (Q12) = {approx/(1<<Q):.6f}")
    print(f"    真值     = {truth:.3f} (Q12) = {truth/(1<<Q):.6f}")
    print(f"    相对误差 = {rel*100:+.4f} %")

    # ---------- 2. 全范围误差统计 ----------
    print("\n[2] 全范围扫描 f_byte = 0..255")
    max_abs, max_rel, worst_fb = 0.0, 0.0, 0
    sum_abs = 0.0
    for fb in range(256):
        truth = 2.0 ** (fb / 256.0)
        approx = exp2_lookup_lerp(fb, table) / (1 << Q)
        abs_err = abs(approx - truth)
        rel_err = abs_err / truth
        sum_abs += abs_err
        if abs_err > max_abs:
            max_abs, max_rel, worst_fb = abs_err, rel_err, fb
    print(f"    max_abs_err = {max_abs:.6f}")
    print(f"    max_rel_err = {max_rel*100:.4f} %")
    print(f"    平均绝对误差 = {sum_abs/256:.6f}")
    print(f"    最坏点 f_byte={worst_fb} (f={worst_fb/256:.4f})，段内位置 {worst_fb % 32}/32")

    # ---------- 3. 误差分布形态 ----------
    print("\n[3] 误差分布：8 段的端点误差 vs 段中误差（验证'段中最大'）")
    print(f"{'段':>3} {'x范围':>12} {'端点误差(理论0)':>14} {'段中误差':>10} {'段内最大':>10}")
    for seg in range(8):
        x0, x1 = seg / 8, (seg + 1) / 8
        # 端点处误差（只含表项舍入，应≈0）
        e_lo = abs(exp2_lookup_lerp(seg * 32, table) / (1 << Q) - 2.0 ** x0)
        # 段中误差（线性化误差最大处）
        xm = (x0 + x1) / 2
        fb_mid = int(round(xm * 256))
        e_mid = abs(exp2_lookup_lerp(fb_mid, table) / (1 << Q) - 2.0 ** xm)
        # 段内最大（扫描 32 级）
        emax_seg = 0.0
        for j in range(32):
            fb_j = seg * 32 + j
            approx_j = exp2_lookup_lerp(fb_j, table) / (1 << Q)
            truth_j = 2.0 ** (fb_j / 256)
            if abs(approx_j - truth_j) > emax_seg:
                emax_seg = abs(approx_j - truth_j)
        print(f"{seg:>3} [{x0:.3f},{x1:.3f}) {e_lo:>14.2e} {e_mid:>10.2e} {emax_seg:>10.2e}")
    print("    → 端点误差≈0（表项精确采样），段中误差≈段内最大 → 误差确由'弦vs曲线'主导")

    # ---------- 4. 三个方案对照 ----------
    print("\n[4] 方案对照（表大小 vs 精度 vs 计算量）")
    # 方案A：纯查表 8 段，不插值（段内一律取左端点）
    errA = []
    for fb in range(256):
        truth = 2.0 ** (fb / 256.0)
        approx = table[fb >> FRAC_BITS] / (1 << Q)
        errA.append(abs(approx - truth))
    # 方案B：9 端点 + 5bit 插值（本方案）
    errB = []
    for fb in range(256):
        truth = 2.0 ** (fb / 256.0)
        approx = exp2_lookup_lerp(fb, table) / (1 << Q)
        errB.append(abs(approx - truth))
    # 方案C：256 项全表（无插值，f 直接查 1/256 网格）
    full = gen_exp2_table(segs=256)
    errC = []
    for fb in range(256):
        truth = 2.0 ** (fb / 256.0)
        approx = full[fb] / (1 << Q)     # 端点对齐，误差 = 舍入误差
        errC.append(abs(approx - truth))
    print(f"    A 纯8段查表(不插值)  : 表9项   max_abs={max(errA):.6f}  ← 段内误差大")
    print(f"    B 9端点+5bit插值(本) : 表9项   max_abs={max(errB):.6f}")
    print(f"    C 256项全表          : 表257项 max_abs={max(errC):.6f}")
    print(f"\n    B 比 A 误差小 {max(errA)/max(errB):.0f} 倍 —— 插值几乎免费地换来了精度")
    print(f"    B 和 C 精度同级，但表只有 C 的 {257/9:.0f}/1 之一 —— 用两次整数乘加换 247 个表项")

    print("\n小结: 3bit+5bit 配置下 2^f 的近似误差 ≈ 1.8e-3（相对 0.1% 量级），")
    print("主导误差是'弦 vs 曲线'的线性化误差。下一 step 把它接进完整 softmax。")


if __name__ == "__main__":
    main()
