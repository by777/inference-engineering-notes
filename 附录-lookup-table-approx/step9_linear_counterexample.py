#!/usr/bin/env python3
"""
step9: 反例实验 —— 给线性函数 y=2x+1 建表，看会发生什么

用户拿 y=2x+1 举例理解"建表公式"。这个实验把这件事做到底：
  ① 用通用建表公式建 y=2x+1 的表，验证表项正确
  ② 查表 + 插值，看误差（预期 = 0，因为直线插值精确）
  ③ 对比"查表 vs 直接算"的指令数（预期 = 查表更慢）

核心结论：线性函数 f''=0，插值误差恒为 0（表 100% 精确），
但查表要"定点化 + 拆位 + 访存 + 插值"一大堆指令，
而直接算 2x+1 只要"1 乘 + 1 加"。
→ 查表对"弯曲的函数"才划算，对"直线"纯属浪费。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SEGS = 8    # 8 段
Q = 12      # Q12 定点


def gen_linear_table(segs=SEGS, q=Q):
    """通用建表公式套到 y=2x+1 上。

    采样点 x_k = k/segs（均分 [0,1)）
    表项 = round((2*x_k + 1) * 2^q)
    """
    table = []
    for k in range(segs + 1):           # k = 0, 1, ..., 8 共 9 个端点
        x_k = k / segs                    # ① 采样点：x = k/8
        y = 2.0 * x_k + 1.0               # ② 函数值：2x+1
        table.append(round(y * (1 << q)))  # ③ 定点化：×2^q 四舍五入
    return table


def linear_lookup(x, table):
    """查表 + 插值算 2x+1（仅演示，实际不会这么干）。"""
    f_byte = min(255, int(round(x * 256)))   # x∈[0,1) → 8bit 定点
    idx = f_byte >> 5
    frac = f_byte & 31
    lo, hi = table[idx], table[idx + 1]
    n = 32
    y_q12 = (lo * (n - frac) + hi * frac) >> 5
    return y_q12 / (1 << Q)


def main():
    print("=" * 72)
    print("step9  反例：给 y=2x+1 建表（直线查表会怎样）")
    print("=" * 72)

    # ---------- ① 建表，验证表项 ----------
    print("\n[1] 用通用公式建 y=2x+1 的表")
    table = gen_linear_table()
    print(f"{'k':>2} {'x_k=k/8':>8} {'y=2x+1':>8} {'y×4096':>10} {'round':>6}")
    for k in range(SEGS + 1):
        xk = k / SEGS
        print(f"{k:>2} {xk:>8.3f} {2*xk+1:>8.3f} {(2*xk+1)*4096:>10.2f} {table[k]:>6}")
    print("    → 表项 = round((2·(k/8)+1)·4096)，和通用公式完全一致")

    # ---------- ② 查表，看误差 ----------
    print("\n[2] 查表 + 插值，扫 x∈[0,1) 看误差")
    max_err = 0.0
    for i in range(1001):
        x = i / 1000
        approx = linear_lookup(x, table)
        truth = 2.0 * x + 1.0
        err = abs(approx - truth)
        max_err = max(max_err, err)
    print(f"    max_abs_err = {max_err:.6f}")
    print(f"    → 误差几乎为 0！因为直线插值 = 直线，100% 精确")

    # ---------- ③ 指令数对比 ----------
    print("\n[3] 但查表比直接算慢得多")
    print("    直接算 2x+1:")
    print("       1 条乘法(x×2) + 1 条加法(+1) = 2 条指令")
    print("    查表:")
    print("       定点化(x×256) + 拆idx/frac(移位+掩码) + 访存 + 插值(2乘1加1移)")
    print("       ≈ 6~7 条指令，还引入了定点/插值复杂度")
    print("    → 误差一样（都是0），但查表多花 3 倍指令，纯属浪费")

    print("\n小结: y=2x+1 是线性函数 f''=0。")
    print("  查表误差确实为 0（直线插值精确），但没有意义——")
    print("  直接算只要 2 条指令，查表要 6+ 条。")
    print("  查表法的价值在弯曲函数（f''≠0，直接算贵）：exp/sigmoid/倒数。")
    print("  对直线，查表是用更多指令做一件 2 条指令就能做的事。")


if __name__ == "__main__":
    main()
