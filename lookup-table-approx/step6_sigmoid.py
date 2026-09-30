#!/usr/bin/env python3
"""
step6: sigmoid 查表 —— 对称性砍半表 + 输入区间对比

sigmoid(x) = 1/(1+e^-x)

和 exp 的对比（这是本 step 的核心教学点）：
  · exp 有天然 range reduction（2^n·2^f），输入压到 [0,1)，8 段就够
  · sigmoid 没有这种压缩，但有两个别的性质：
      ① 对称性 σ(-x) = 1 - σ(x)  → 只需查 x≥0，表砍半
      ② 饱和性 |x|>8 后 σ→1      → 有效区间 [0,8]

本 step 展示：
  ① 对称性验证（数学恒等，零误差）
  ② 直接对 [0,8] 建表（9 端点，x=0..8）
  ③ 全范围误差扫描 + 对拍浮点
  ④ 对比 exp：同样 8 段，sigmoid 误差为何大很多（输入区间差 8 倍 → h² 差 64 倍）
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lut_common import FRAC_BITS, Q, gen_exp2_table, exp2_lookup_lerp

X_MAX = 8.0          # 饱和阈值：x>8 后 σ 与 1 的差 < 4e-4，直接返回 1
# X_MAX 怎么定的（精度预算反推，不是看图像估的）：
#   截断误差 = 1 - σ(x) ≈ e^-x（x 较大时）
#   要求截断误差 ≤ 插值误差(≈1.3e-2)的 1/10 = 1.3e-3
#   → e^-x < 1.3e-3  →  x > ln(770) ≈ 6.6
#   → 取整 + 留余量 = 8（此时误差 3.3e-4，比 1.3e-3 再小 4 倍）
SEGS = 8             # [0,8] 均匀 8 段，段宽 h=1


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def gen_sigmoid_table(segs=SEGS, q=Q):
    """对 x∈[0,8] 均匀 segs 段建表，端点 x_k = k·(X_MAX/segs)。
    表项 = round(σ(x_k)·2^q)。共 segs+1 项。"""
    h = X_MAX / segs                   # 段宽
    scale = 1 << q                     # 定点标尺 2^q

    table = []
    for k in range(segs + 1):          # 8 段 = 9 个端点
        x_k = k * h                    # 第 k 个结点的 x 值
        value = sigmoid(x_k)           # 该点的函数真值 σ(x_k)
        table.append(round(value * scale))   # 转定点 + 四舍五入
    return table


def sigmoid_lookup(x, table):
    """查表算 σ(x)（Q12 定点输出，返回真值 float 便于对拍）。

    分三路：
      x < 0    → 对称性 σ(-x)=1-σ(x)
      0≤x≤8    → 查表 + 插值
      x > 8    → 饱和，返回 1
    """
    if x < 0:
        return 1.0 - sigmoid_lookup(-x, table)      # 对称性
    if x > X_MAX:
        return 1.0                                   # 饱和
    # 查表 + 插值（x→8bit 定点，覆盖 [0,8]）
    f_byte = min(255, int(round(x / X_MAX * 256)))
    idx = f_byte >> FRAC_BITS
    frac = f_byte & ((1 << FRAC_BITS) - 1)
    lo, hi = table[idx], table[idx + 1]
    n = 1 << FRAC_BITS
    val_q12 = (lo * (n - frac) + hi * frac) >> FRAC_BITS   # Q12 定点插值
    return val_q12 / (1 << Q)                              # ÷4096 还原真值


def main():
    print("=" * 72)
    print("step6  sigmoid 查表：对称性 + 输入区间对比")
    print("=" * 72)

    # ---------- ① 对称性验证 ----------
    print("\n[1] 对称性验证: σ(-x) = 1 - σ(x)  （数学恒等，零误差）")
    for x in [0.5, 1.0, 3.0, 8.0]:
        lhs = sigmoid(-x)
        rhs = 1.0 - sigmoid(x)
        print(f"    x={x:<4}  σ(-x)={lhs:.6f}  1-σ(x)={rhs:.6f}  {'✓' if abs(lhs-rhs)<1e-15 else '✗'}")

    # ---------- ② 建表 ----------
    print("\n[2] 对 [0,8] 建表（8 段，段宽 h=1）")
    table = gen_sigmoid_table()
    print(f"{'k':>2} {'x_k':>6} {'σ(x_k)':>10} {'σ×4096':>10} {'round':>6}")
    for k in range(SEGS + 1):
        xk = k * (X_MAX / SEGS)
        print(f"{k:>2} {xk:>6.1f} {sigmoid(xk):>10.6f} {sigmoid(xk)*4096:>10.2f} {table[k]:>6}")

    # ---------- ③ 全范围扫描 + 对拍 ----------
    print("\n[3] 全范围扫描 x∈[-16,16] 对拍浮点")
    max_abs, worst_x = 0.0, 0.0
    for i in range(3201):
        x = -16.0 + i * (32.0 / 3200)
        approx = sigmoid_lookup(x, table)
        truth = sigmoid(x)
        err = abs(approx - truth)
        if err > max_abs:
            max_abs, worst_x = err, x
    print(f"    max_abs_err = {max_abs:.6f}  @ x={worst_x:.3f}")
    print(f"    注：x>8 走饱和(误差<4e-4)，x<0 走对称(误差同正半轴)")

    # ---------- ④ 对比 exp ----------
    print("\n[4] 对比 exp：误差 ∝ h²·max|f''|，两个量都要看")
    # exp 和 sigmoid 的 max|f''| 不同（这是 [4] 的关键，不能只比段宽）
    f2pp_exp = (math.log(2) ** 2) * 2.0          # exp 二阶导，x=1 处 = 0.961
    f2pp_sig = 0.0962                             # sigmoid 二阶导 max，x≈1.317 处
    h_exp, h_sig = 1.0 / 8, 1.0                   # 段宽
    print(f"    exp      h={h_exp:.4f}  max|f''|={f2pp_exp:.3f}  → ε≈{h_exp**2/8*f2pp_exp:.4f}")
    print(f"    sigmoid  h={h_sig:.4f}  max|f''|={f2pp_sig:.3f}  → ε≈{h_sig**2/8*f2pp_sig:.4f}")
    print(f"    拆解: 段宽差 8 倍 → h² 贡献 {(h_sig/h_exp)**2:.0f} 倍")
    print(f"          但 sigmoid 曲率只有 exp 的 {f2pp_sig/f2pp_exp:.1f}/1 → 抵消")
    print(f"          净效果 ≈ {h_sig/h_exp:,.0f}² × {f2pp_sig/f2pp_exp:.2f} = {(h_sig/h_exp)**2*f2pp_sig/f2pp_exp:.2f} 倍")
    print(f"    实测 max_err: sigmoid {max_abs:.4f} / exp 0.0018 ≈ {max_abs/0.0018:.1f} 倍（与 6.4 同量级）")
    print(f"    → 结论：误差 ∝ 段宽² × 曲率。段宽是主因（平方），曲率是修正项")

    print("\n小结: sigmoid 靠对称性砍半表（只查 x≥0）、饱和性截断区间（[0,8]）。")
    print("同 8 段下误差比 exp 大 ~6 倍（段宽平方×曲率），")
    print("想补精度要么加段数、要么非均匀分段。")


if __name__ == "__main__":
    main()
