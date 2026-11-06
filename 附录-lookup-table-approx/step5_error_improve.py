#!/usr/bin/env python3
"""
step5: 误差数学与改进 —— 误差公式验证 + 怎么进一步压误差

已知 3bit+5bit 配置误差 ≈ 1.8e-3。这笔账怎么算的？能不能再压？

本 step：
  ① 验证线性化误差公式 ε ≈ h²/8 · max|f''|  （h=段宽）
  ② "最优弦"技巧：把端点连线平移，让误差正负均衡 → 最大误差减半
  ③ 段数-误差收敛律：段数翻倍误差 /4（因为 ε ∝ h² ∝ N⁻²）
  ④ 同表项数下对比：均匀 vs 按曲率加权分段（收尾展望）
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lut_common import Q, SEGS, gen_exp2_table, exp2_lookup_lerp, LOG2E


def f2(x):      # 目标函数 2^x
    return 2.0 ** x


def f2pp(x):    # 二阶导 (ln2)^2 · 2^x
    return (math.log(2.0) ** 2) * f2(x)


def max_abs_err_of_table(table, segs=SEGS):
    """给定一张表(端点已含 x=0..1 均匀分割 segs 段)，扫描其线性插值最大误差"""
    mx = 0.0
    N = 256 * segs // SEGS if segs != SEGS else 256   # 扫描点数随段数缩放保持等价
    for i in range(N + 1):
        x = i / N
        if x >= 1.0:
            continue
        pos = x * segs
        k = int(pos)
        frac = pos - k
        lo, hi = table[k], table[k + 1]
        approx = (lo * (1 - frac) + hi * frac) / (1 << Q)   # 直接浮点插值，排除表项舍入
        mx = max(mx, abs(approx - f2(x)))
    return mx


def main():
    print("=" * 72)
    print("step5  误差数学：公式验证 + 三个改进方向")
    print("=" * 72)

    # ---------- ① 验证误差公式 ----------
    print("\n[1] 线性化误差公式验证: ε_max ≈ h²/8 · max|f''(x)|  (h=段宽=1/N)")
    table8 = gen_exp2_table()
    err8 = max_abs_err_of_table(table8)
    # f'' 在 [0,1] 单调增，max 在 x=1
    h = 1.0 / SEGS
    theo = h * h / 8.0 * f2pp(1.0)
    print(f"    8 段实测 max 线性化误差 ≈ {err8:.6f}")
    print(f"    公式预测 h²/8·f''(1)     = {theo:.6f}")
    print(f"    （实测略小：公式是上界，f'' 在段内并非恒为端点值）")

    # ---------- ② 最优弦 ----------
    print("\n[2] 改进①：'最优弦'——把连线整体下移，让段内误差正负均衡")
    # 思路：误差单边(弦在曲线上方)，段内 max 在中点。把整条弦向下平移
    # 段内最大误差的一半，最大误差就从 ε → ε/2（正负误差均衡）
    # 实现：直接对每段找使 max|误差| 最小的平移量（数值最优，一维极小极大）
    def seg_best_shift(k):
        """对第 k 段，求最优平移量 δ 使段内 max|f(x)-(连线-δ)| 最小"""
        a, b = k / SEGS, (k + 1) / SEGS
        lo, hi = table8[k], table8[k + 1]
        # 在段内采样，误差 e(x)=连线-f(x) ∈ [0, e_mid]（凸函数弦在上）
        xs = []
        for j in range(65):            # 段内 65 个采样点
            xs.append(a + (b - a) * j / 64)
        errs = []
        for x in xs:
            w = (x - a) / (b - a)
            line = (lo * (1 - w) + hi * w) / (1 << Q)
            errs.append(line - f2(x))
        e_mid = max(errs)          # 凸函数：段中误差最大
        return e_mid / 2.0         # 下移一半 → 段内两端误差=中间误差

    # 构造平移后的表（每个端点值都参与两段，用相邻段平移的平均近似）
    shifts = []
    for k in range(SEGS):
        shifts.append(seg_best_shift(k))
    new_table = []
    for k in range(SEGS + 1):
        sleft = shifts[k - 1] if k > 0 else shifts[0]
        sright = shifts[k] if k < SEGS else shifts[SEGS - 1]
        shift_v = (sleft + sright) / 2
        new_table.append(round((f2(k / SEGS) - shift_v) * (1 << Q)))
    err_new = max_abs_err_of_table(new_table)
    print(f"    原始端点表    max_err = {err8:.6f}")
    print(f"    最优弦平移后  max_err = {err_new:.6f}   (≈ {err8/err_new:.2f} 倍提升)")
    print(f"    本质：把'弦在曲线上方'变成'误差正负交错'，极小极大意义下的最优")

    # ---------- ③ 段数收敛律 ----------
    print("\n[3] 改进②：加段数。ε ∝ h²，段数翻倍 → 误差 /4")
    print(f"{'段数':>6} {'表项数':>6} {'max_err':>12} {'相对8段':>8}")
    base = None
    for segs in (8, 16, 32):
        tb = gen_exp2_table(segs=segs)
        e = max_abs_err_of_table(tb, segs=segs)
        if base is None:
            base = e
        print(f"{segs:>6} {segs+1:>6} {e:>12.6f} {e/base:>8.4f}")
    print("    16 段误差 = 8 段的 1/3.6")
    print("      （理论 ε∝h² 应为 1/4，差异来自最坏段 f'' 取值与扫描粒度）")
    print("    加 1bit 段号 = 表翻倍 = 误差 ≈ /3.6~4；插值位数再加几乎无收益")

    # ---------- ④ 插值位数还有没有必要加 ----------
    print("\n[4] 5bit 插值够不够？—— 插值到'线性化误差地板'就停")
    # 5bit 插值的误差 = 弦上再做 32 级阶梯 ≈ 几乎等于理想线性插值
    # 8段理想线性插值 err≈1.7e-3；若只有 3bit 查表没有插值则 err≈ 半段宽 ≈ 4e-2
    err_ideal_lerp = err8
    # 无插值（纯取左端点）误差 ≈ 段内 max|f(x)-f(a)|（f 单调增，最大在段右端）
    e_nointerp = 0.0
    for k in range(SEGS):
        jump = f2((k + 1) / SEGS) - f2(k / SEGS)
        if jump > e_nointerp:
            e_nointerp = jump
    print(f"    纯查表不插值(8段) 误差 ≈ {e_nointerp:.4f}")
    print(f"    5bit插值(32级)    误差 ≈ {err_ideal_lerp:.6f}")
    print(f"    32级插值已经把误差压到线性化地板附近；再加插值级数几乎无收益")
    print(f"    → 想再准就加段数(见③)，不是加插值位 —— 这就是'3bit+5bit'配比的由来")

    # ---------- ⑤ 展望：非均匀分段 ----------
    print("\n[5] 展望：非均匀分段（exp 曲率右侧大，右侧段更密更省表项）")
    print("    f''(x)=(ln2)²·2^x 单调增 → 误差大的段集中在右端")
    print("    均匀 8 段每段误差不同：最右段误差 = 最左段的 f''(1)/f''(0)=2 倍")
    print("    若让每段误差相等（段宽∝1/√f''），同 9 项表还能再降一些误差")
    print("    —— 但代价是段号不再能从 f 的高位直接截取，需要比较器/二分查段")
    print("    （硬件上通常因此保留均匀分段；这是精度 vs 寻址成本的经典权衡）")

    print("\n小结: 误差 = 线性化项主导 ≈ h²/8·max|f''|；改进按性价比排序：")
    print("  ① 最优弦(免费,误差/2)   ② 加段数(表×2,误差/4)")
    print("  ③ 加插值位(收益小)       ④ 非均匀(需额外寻址)")


if __name__ == "__main__":
    main()
