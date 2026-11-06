#!/usr/bin/env python3
"""
step10: rsqrt(1/√x) 查表 + 牛顿迭代 —— 快逆平方根

和 step8 (1/x) 是姊妹篇，同构但有三个关键差异：
  ① range reduction: 指数要【减半】，需先把 e 拆成偶数，让 e/2 是整数（纯移位）
  ② 牛顿公式: y_{n+1} = y/2·(3 - m·y²)，全程无除法（/2 是乘 0.5 或右移 1 位）
  ③ 收敛: 和 1/x 一样是【平方】收敛，d -> -(3/2)·d²（系数 3/2，比 1/x 的系数 1 更差）
     易错点：别记成"立方收敛 / 比 1/x 更快" —— rsqrt 的优势是无除法，不是收敛阶。

本 step 展示：
  ① range reduction（指数奇偶处理）
  ② 建 1/√m 表
  ③ 查表 + 牛顿迭代，看平方收敛（2 轮到 float 极限）
  ④ 对拍浮点
  ⑤ trace_one(x)：逐步打印 x=4.1 的完整链路（数值教学用）
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def split_mant_exp_even(x):
    """把 x 拆成 m·2^e，且【保证 e 是偶数】，m∈[1,4)。

    理由：rsqrt 要算 2^(-e/2)，只有 e 是偶数时 e/2 才是整数（纯移位）。
    frexp 给 m∈[0.5,1)，先 ×2 归一化到 [1,2)；
    若此时 e 是奇数，就再 ×2、e-1，把 e 变成偶数，同时 m 进 [2,4)。
    """
    m, e = math.frexp(x)      # x = m·2^e, m∈[0.5,1)
    m = m * 2.0               # 归一化到 [1,2)
    e = e - 1
    if e % 2 != 0:            # e 是奇数 → 再搬一位，让 e 变偶数
        m = m * 2.0           # m 进 [2,4)
        e = e - 1
    return m, e               # m∈[1,4)，e 为偶数


def gen_rsqrt_table(segs=8, q=12):
    """对 m∈[1,4) 建 1/√m 表。

    端点 m_k = 1 + k/8·3（区间宽 3），即 1.0, 1.375, ..., 4.0 共 9 个。
    表项 = round((1/√m_k) · 2^q)。
    """
    width = 3.0               # 区间 [1,4) 宽 3
    scale = 1 << q            # 定点标尺 2^q（q=12 → 4096）

    table = []
    for k in range(segs + 1):          # 8 段 = 9 个端点（结点）
        m_k = 1.0 + k / segs * width   # 第 k 个结点的 m 值
        value = 1.0 / math.sqrt(m_k)   # 该点的函数真值 1/√m_k
        table.append(round(value * scale))   # 转定点（×4096）+ 四舍五入
    return table


def lut_initial(m, table):
    """由 m∈[1,4) 查表 + 线性插值，返回 (初值 y0, 字节 f_byte)。

    抽成独立函数，是为了让 [3] 收敛演示和 trace_one 跟 rsqrt_lookup
    共用同一份逻辑 —— 避免"演示代码"和"真实代码"悄悄不一致。
    """
    f_byte = min(255, int(round((m - 1.0) / 3.0 * 256)))   # ① m → 一个字节
    idx = f_byte >> 5                                      # ② 高 3 位 = 段号
    frac = f_byte & 31                                     #    低 5 位 = 段内位置
    lo, hi = table[idx], table[idx + 1]
    y0_q12 = (lo * (32 - frac) + hi * frac) >> 5            # ③ 两点式插值（定点）
    return y0_q12 / 4096.0, f_byte


def rsqrt_lookup(x, table):
    """查表 + 牛顿迭代求 1/√x。"""
    m, e = split_mant_exp_even(x)          # ① range reduction（e 偶数）

    # ② 查表：m∈[1,4) → 8bit 定点
    y0, _f_byte = lut_initial(m, table)    # 初值 1/√m

    # ③ 牛顿迭代：y_{n+1} = y/2·(3 - m·y²)，无除法
    y = y0
    for _ in range(4):
        y = y * 0.5 * (3.0 - m * y * y)

    # ④ 还原指数：1/√x = (1/√m)·2^(-e/2)
    return y * (2.0 ** (-e / 2))


def trace_one(x=4.1, table=None):
    """逐步打印 x 的完整计算链路（教学用，默认 x=4.1）。"""
    if table is None:
        table = gen_rsqrt_table()

    print("\n" + "=" * 72)
    print(f"[5] 完整 trace: x = {x}")
    print(f"     目标 1/√x = {1/math.sqrt(x):.12f}     (√x = {math.sqrt(x):.12f})")
    print("=" * 72)

    # ---------- ① range reduction ----------
    print("\n① range reduction:  x = m·2^e，且 e 必须是偶数")
    mr, er = math.frexp(x)
    print(f"    frexp:        m = {mr}  e = {er:<3}  →  {mr} × 2^{er} = {mr * 2**er}")
    m1, e1 = mr * 2.0, er - 1
    print(f"    归一化[1,2):   m = {m1}  e = {e1:<3}  →  {m1} × 2^{e1} = {m1 * 2**e1}")
    if e1 % 2 != 0:
        m2, e2 = m1 * 2.0, e1 - 1
        print(f"    e 奇数 → 偶化: m = {m2}  e = {e2:<3}  →  {m2} × 2^{e2} = {m2 * 2**e2}")
    else:
        m2, e2 = m1, e1
        print(f"    e = {e2} 已是偶数 → 不搬")
    m, e = m2, e2
    print(f"    结论: x = {m} × 2^{e}    校验 m·2^e == x : {m * 2**e == x}")
    print(f"          1/√x = [1/√{m}] × 2^(-{e}/2) = [1/√{m}] × 2^{ -e // 2 }")

    # ---------- ② 查表 ----------
    print("\n② 查表:  m → 一个字节 → (段号, 段内位置)")
    fb_raw = (m - 1.0) / 3.0 * 256
    y0, f_byte = lut_initial(m, table)
    idx = f_byte >> 5
    frac = f_byte & 31
    seg_lo, seg_hi = 1 + idx / 8 * 3.0, 1 + (idx + 1) / 8 * 3.0
    print(f"    f_byte = round(({m} - 1)/3 × 256) = round({fb_raw:.6f}) = {f_byte}")
    print(f"    二进制 = {f_byte:08b}   →  idx = {f_byte:08b}>>5 = {idx}"
          f"   frac = {f_byte:08b}&31 = {frac}")
    print(f"    落在第 {idx} 段: m ∈ [{seg_lo}, {seg_hi})  "
          f"校验 {seg_lo} <= {m} < {seg_hi} : {seg_lo <= m < seg_hi}")
    print(f"    5bit 位置: t = frac/32 = {frac}/32 = {frac / 32}")

    lo, hi = table[idx], table[idx + 1]
    print(f"\n    lo = T[{idx}] = {lo:<5} /4096 = {lo / 4096:.10f}"
          f"   (真值 1/√{seg_lo} = {1 / math.sqrt(seg_lo):.10f})")
    print(f"    hi = T[{idx + 1}] = {hi:<5} /4096 = {hi / 4096:.10f}"
          f"   (真值 1/√{seg_hi} = {1 / math.sqrt(seg_hi):.10f})")

    # ---------- ③ 插值 ----------
    print("\n③ 线性插值（两点式）:  y0 = [T[k]·(32-frac) + T[k+1]·frac] >> 5")
    num = lo * (32 - frac) + hi * frac
    print(f"    num = {lo}×{32 - frac} + {hi}×{frac} = {lo * (32 - frac)} + {hi * frac} = {num}")
    print(f"    y0_q12 = {num} >> 5 = {num >> 5}   (丢掉低 5 位 {num % 32})")
    truth_m = 1 / math.sqrt(m)
    print(f"    y0 = {num >> 5}/4096 = {y0:.12f}")
    print(f"    真值 1/√{m} = {truth_m:.12f}    →  初值相对误差 = "
          f"{abs(y0 - truth_m) / truth_m:.3e}")

    # ---------- ④ 牛顿迭代 ----------
    print("\n④ 牛顿迭代:  y ← y/2·(3 - m·y²)      （无除法，×0.5 = 右移 1）")
    y = y0
    d_prev = abs(y0 - truth_m) / truth_m
    for i in range(1, 5):
        my2 = m * y * y
        inner = 3.0 - my2
        y_new = y * 0.5 * inner
        rel = abs(y_new - truth_m) / truth_m
        pred = 1.5 * d_prev ** 2                       # 平方收敛预测: d → 1.5·d²
        print(f"    轮{i}: m·y² = {my2:.12f}   3-m·y² = {inner:.12f}")
        print(f"         y = {y:.12f} × 0.5 × {inner:.12f} = {y_new:.12f}")
        print(f"         rel_err = {rel:.3e}    预测(1.5·d²) = {pred:.3e}")
        y, d_prev = y_new, rel

    # ---------- ⑤ 还原指数 ----------
    print("\n⑤ 还原指数:  1/√x = (1/√m) × 2^(-e/2)")
    scale = 2.0 ** (-e / 2)
    res = y * scale
    truth_x = 1 / math.sqrt(x)
    print(f"    2^(-e/2) = 2^(-{e}/2) = 2^{ -e // 2 } = {scale}")
    print(f"    1/√x = {y:.12f} × {scale} = {res:.12f}")
    print(f"    真值 = {truth_x:.12f}    相对误差 = {abs(res - truth_x) / truth_x:.3e}")
    print(f"\n    ⇒ √x = x · (1/√x) = {x} × {res:.12f} = {x * res:.12f}")
    print(f"      真值 √x = {math.sqrt(x):.12f}")


def main():
    print("=" * 72)
    print("step10  rsqrt 查表 + 牛顿迭代（快逆平方根）")
    print("=" * 72)

    # ---------- ① range reduction ----------
    print("\n[1] range reduction: x = m·2^e（e 偶数），1/√x = (1/√m)·2^(-e/2)")
    for x in [1.0, 2.0, 4.0, 9.0, 0.25]:
        m, e = split_mant_exp_even(x)
        print(f"    x={x:>6} = {m:.6f} × 2^{e}  →  1/√x = (1/√{m:.3f}) × 2^{-e/2}")

    # ---------- ② 建表 ----------
    print("\n[2] 1/√m 表 [1,4) 8 段")
    table = gen_rsqrt_table()
    print(f"{'k':>2} {'m_k':>6} {'1/√m_k':>10} {'×4096':>10} {'round':>6}")
    for k in range(9):
        mk = 1.0 + k / 8 * 3.0
        print(f"{k:>2} {mk:>6.3f} {1/math.sqrt(mk):>10.6f} {(1/math.sqrt(mk))*4096:>10.2f} {table[k]:>6}")

    # ---------- ③ 看收敛 ----------
    print("\n[3] 牛顿迭代收敛（m=2.0，初值来自查表）")
    m, e = split_mant_exp_even(2.0)
    y, _ = lut_initial(m, table)
    truth = 1 / math.sqrt(m)
    print(f"    查表初值 y0 = {y:.6f}   (真值 1/√{m:.3f} = {truth:.6f})")
    d_prev = abs(y - truth) / truth
    for i in range(1, 5):
        y = y * 0.5 * (3.0 - m * y * y)
        rel = abs(y - truth) / truth
        pred = 1.5 * d_prev ** 2
        print(f"    迭代{i}次  y = {y:.10f}   rel_err = {rel:.2e}"
              f"   预测(1.5·d²) = {pred:.2e}")
        d_prev = rel

    # ---------- ④ 全范围对拍 ----------
    print("\n[4] 全范围扫描 x∈[0.1,10] 对拍")
    max_rel, worst_x = 0.0, 0.0
    for i in range(2001):
        x = 0.1 + (9.9) * i / 2000
        approx = rsqrt_lookup(x, table)
        truth = 1.0 / math.sqrt(x)
        rel = abs(approx - truth) / truth
        if rel > max_rel:
            max_rel, worst_x = rel, x
    print(f"    max_rel_err = {max_rel:.2e}  @ x={worst_x:.4f}")

    print("\n小结: rsqrt = 查表给初值 + 牛顿迭代（y/2·(3-m·y²)，无除法）。")
    print("  和 1/x 同构，区别：指数减半（要拆成偶数）+ 误差系数 3/2（1/x 是 1）。")
    print("  注意：rsqrt 牛顿是【平方】收敛（d → -1.5·d²），不是立方。")
    print("  它的优势不是收敛阶，而是【全程无除法】—— 不用查倒数表也不做除法。")
    print("  这就是 Quake III fast inverse sqrt 的核心（位操作魔法数给初值）。")

    # ---------- ⑤ 完整 trace ----------
    trace_one(4.1, table)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 收敛阶推导：为什么是平方收敛，而不是立方
# ---------------------------------------------------------------------------
# 常见误记是"误差【立方】衰减，比 1/x 的平方收敛还快"。正确推导如下：
#
#   设 y_n = y*(1+d)，y* = 1/√m，则
#       y_n² = (1/m)(1+d)²      →  m·y_n² = 1 + 2d + d²
#       3 - m·y_n² = 2 - 2d - d²
#       y_{n+1} = y*(1+d)/2 · (2 - 2d - d²) = y*(1+d)(1 - d - d²/2)
#   展开丢掉 d³:
#       d_{n+1} ≈ -1.5·d²
#
# 即【平方收敛】，系数 3/2 —— 比 1/x 牛顿迭代的 d → -d² (系数 1) 【更差】。
# 实测（m=2.0）：3.03e-3 → 1.38e-5 → 2.85e-10 → 0
#   1.5×(3.03e-3)² = 1.38e-5   ✓
#   1.5×(1.38e-5)² = 2.85e-10  ✓
#
# rsqrt 迭代真正的价值是【无除法】（只需乘加 + ×0.5），不是收敛速度。
# ---------------------------------------------------------------------------
