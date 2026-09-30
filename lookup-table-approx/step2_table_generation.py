#!/usr/bin/env python3
"""
step2: 表项是怎么生成的 —— 表不是"查"出来的，是"离线算好"的

三要素：
  1. 目标函数      g(f) = 2^f,  f∈[0,1)      （来自 step1 的 range reduction）
  2. 采样方式      均匀切 8 段 → 9 个端点 x_k = k/8
  3. 定点格式      Q12：表项 = round(g(x_k) · 2^Q)

关键认知：
  · 建表发生在"离线/上位机"，用浮点怎么算都行，算完转整数
  · 板上运行时永远只做 整数移位 + 查表 + 整数乘加，无 FPU 也无所谓
  · 表项只存端点（breakpoint），段内靠 step3 的插值补全 —— 所以表极小
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lut_common import Q, SEGS, gen_exp2_table, table_to_c


def main():
    print("=" * 72)
    print("step2  表项来源: table[k] = round(2^(k/8) · 2^Q)")
    print("=" * 72)

    # ---------- 1. 逐步推导：9 个端点一个个人工算一遍 ----------
    print("\n[1] 手工推导过程（Q12, 即乘 4096 后取整）")
    print(f"{'k':>2} {'端点x=k/8':>10} {'真值 2^x':>12} {'真值×4096':>12} {'round()':>8}")
    for k in range(SEGS + 1):
        xk = k / SEGS
        truth = 2.0 ** xk
        scaled = truth * (1 << Q)
        print(f"{k:>2} {xk:>10.3f} {truth:>12.8f} {scaled:>12.4f} {round(scaled):>8d}")

    # ---------- 2. 实际生成 + 观察规律 ----------
    table = gen_exp2_table()
    print(f"\n[2] gen_exp2_table() 直接生成:")
    print(f"    {table}")
    print(f"    表项数 = {SEGS}+1 = {len(table)}（8 段共享端点）")
    ratios = []
    for i in range(len(table) - 1):
        ratios.append(table[i + 1] / table[i])

    ratio_parts = []
    for r in ratios:
        ratio_parts.append(f"{r:.4f}")
    print(f"    相邻比值 ≈ {' '.join(ratio_parts)}")
    print("    → 比值≈1.0905 = 2^(1/8)。这是 2^x 的固有性质，不是巧合。")

    # ---------- 3. Q 的选择如何影响表项与误差 ----------
    print("\n[3] 定点位数 Q 的影响（决定表项舍入误差）")
    print(f"{'Q':>4} {'scale':>8} {'表项[0..8]':>42} {'max量化误差':>12}")
    for q in (8, 12, 16):
        tb = gen_exp2_table(q=q)
        # 量化误差：round 最多偏 0.5 LSB
        max_qerr = 0.5 / (1 << q)
        parts = []
        for v in tb:
            parts.append(str(v))
        s = " ".join(parts)
        print(f"{q:>4} {1<<q:>8} {s:>42} {max_qerr:.2e}")
    print("    Q12: 表项 ≤ 8192，int16 装得下（18 字节）；Q16 要 int32（36 字节）。")
    print("    取舍：Q 越高舍入误差越小，但表项越宽、乘加中间位越宽。")

    # ---------- 4. 导出 C 数组（为将来接 C/板子代码准备） ----------
    print("\n[4] 导出 C 静态数组（9 × int16 = 18 字节，可放片上 SRAM/寄存器）")
    print(table_to_c(table))

    print("\n小结: 表项 = 离线对 2^x 在 k/8 处采样并定点化。板上只有 9 个 int16，")
    print("没有秘密 —— 精度是'离线算 + 定点化'给的，速度是'板上只查表'给的。")


if __name__ == "__main__":
    main()
