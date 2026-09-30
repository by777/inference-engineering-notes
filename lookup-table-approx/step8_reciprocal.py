#!/usr/bin/env python3
"""
step8: 1/x 查倒数表 + 牛顿迭代 —— "查表 + 迭代混合"

和 exp/sigmoid/tanh 的"纯插值一步到位"不同，1/x 是【查表给初值 + 牛顿迭代收敛】。

════════════════ 原理两步 ════════════════

① range reduction（把无穷输入压到一个小区间）
   任意正数 x 都能写成  x = m · 2^e，其中 m∈[1,2)
   （m 叫尾数 mantissa，e 叫指数 exponent，这就是 float 在内存里的存储格式）
   于是 1/x = 1/(m·2^e) = (1/m) · 2^-e
   → 只需查 [1,2) 上的倒数（m 的区间固定），指数部分 2^-e 又是移位
   → 无穷多个 x 映射到同一个 m，查一次表覆盖它们全部

② 牛顿迭代（用乘法迭代逼近 1/m，误差平方级衰减）
   求 1/m 等价于求 f(y)=1/y - m = 0 的根
   牛顿法：y_{n+1} = y_n - f(y_n)/f'(y_n)
   代入 f(y)=1/y-m，f'(y)=-1/y²，化简得：
       y_{n+1} = y_n·(2 - m·y_n)      ← 只含乘法和减法，没有除法！
   二次收敛：误差每轮【平方】，1e-4 → 1e-8 → 1e-16，2~3 轮到 float 极限

════════════════ 本 step 展示 ════════════════
  ① range reduction 验证（拆 mantissa/exponent）
  ② 建倒数表 [1,2) 8 段
  ③ 查表 + 牛顿迭代，看每轮误差平方衰减
  ④ 对拍浮点，全范围误差
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def split_mant_exp(x):
    """把 x 拆成 m·2^e，m∈[1,2)。

    底层就是 IEEE 754 的位拆解：
      float32 存的是 [符号1][指数8][尾数23]，值 = 1.尾数 × 2^(指数-127)
      math.frexp 返回 (m, e) 使得 x = m·2^e，但它的 m 是 [0.5, 1)（尾数当纯小数）

    我们要的 m 是 [1,2)（前导 1 显式出现），所以：
      把 m 乘 2（0.5→1、0.99→1.98），指数相应减 1，值不变
    """
    m, e = math.frexp(x)  # x = m·2^e，m∈[0.5,1)
    return m * 2.0, e - 1  # 归一化到 [1,2)：乘 2，指数减 1


def gen_rcp_table(segs=8, q=12):
    """对 m∈[1,2) 建倒数表。

    端点 m_k = 1 + k/8（k=0..8），即 1.0, 1.125, ..., 2.0 共 9 个。
    表项 = round((1/m_k) · 2^q)，即端点处倒数的 Q12 定点值。
    """
    width = 1.0                        # 区间 [1,2) 的宽 = 2 - 1
    scale = 1 << q                     # 定点标尺 2^q

    table = []
    for k in range(segs + 1):          # 8 段 = 9 个端点
        m_k = 1.0 + k / segs * width   # 第 k 个结点的 m 值
        value = 1.0 / m_k              # 该点的函数真值 1/m_k
        table.append(round(value * scale))   # 转定点 + 四舍五入
    return table


def rcp_lookup(x, table):
    """查表 + 牛顿迭代求 1/x。

    分四步：
      ① range reduction：拆 x = m·2^e
      ② 查表插值：m∈[1,2) 定点成 8bit，高3bit定段、低5bit插值，得 1/m 的初值
      ③ 牛顿迭代：y = y·(2 - m·y)，二次收敛提精度
      ④ 还原指数：结果再乘 2^-e（移位）
    """
    m, e = split_mant_exp(x)  # ① range reduction

    # ② 查表：m∈[1,2) → 定点 8bit
    #    (m-1) 把 [1,2) 平移到 [0,1)，再 ×256 放大成 0~255 的整数
    f_byte = min(255, int(round((m - 1.0) * 256)))
    idx = f_byte >> 5  # 高 3 bit → 段号 0..7
    frac = f_byte & 31  # 低 5 bit → 段内 32 级位置
    lo, hi = table[idx], table[idx + 1]  # 段左右端点的表项
    n = 32
    # 线性插值：y0 = (lo·(32-frac) + hi·frac) / 32，结果仍是 Q12
    y0_q12 = (lo * (n - frac) + hi * frac) >> 5
    y0 = y0_q12 / 4096.0  # ÷2^Q 把 Q12 还原成真值，作为迭代初值

    # ③ 牛顿迭代：y_{n+1} = y_n·(2 - m·y_n)
    #    注意迭代变量是 y（逼近 1/m），x 用 m（因为 range reduction 后只查 [1,2)）
    y = y0
    for _ in range(3):
        y = y * (2.0 - m * y)  # 二次收敛：误差每轮平方

    # ④ 还原指数：1/x = (1/m)·2^-e
    return y * (2.0 ** (-e))


def main():
    print("=" * 72)
    print("step8  1/x 查倒数表 + 牛顿迭代")
    print("=" * 72)

    # ---------- ① range reduction ----------
    print("\n[1] range reduction: x = m·2^e, 1/x = (1/m)·2^-e")
    for x in [1.5, 3.0, 12.0, 0.25, 100.0]:
        m, e = split_mant_exp(x)
        print(f"    x={x:>6}  = {m:.6f} × 2^{e}   → 1/x = (1/{m:.3f}) × 2^{-e}")

    # ---------- ② 建表 ----------
    print("\n[2] 倒数表 [1,2) 8 段")
    table = gen_rcp_table()
    print(f"{'k':>2} {'m_k':>6} {'1/m_k':>10} {'×4096':>10} {'round':>6}")
    for k in range(9):
        mk = 1.0 + k / 8
        print(f"{k:>2} {mk:>6.3f} {1/mk:>10.6f} {(1/mk)*4096:>10.2f} {table[k]:>6}")

    # ---------- ③ 看迭代收敛 ----------
    print("\n[3] 牛顿迭代收敛（x=1.5，初值来自查表）")
    x = 1.5
    m, e = split_mant_exp(x)
    f_byte = int(round((m - 1.0) * 256))
    idx = f_byte >> 5
    frac = f_byte & 31
    lo, hi = table[idx], table[idx + 1]
    y = ((lo * (32 - frac) + hi * frac) >> 5) / 4096.0
    truth = 1 / m
    print(f"    查表初值 y0 = {y:.6f}   (真值 1/{m:.3f} = {truth:.6f})")
    for i in range(1, 4):
        y = y * (2.0 - m * y)
        rel = abs(y - truth) / truth
        print(f"    迭代{i}次  y = {y:.10f}   相对误差 = {rel:.2e}")

    # ---------- ④ 全范围对拍 ----------
    print("\n[4] 全范围扫描 x∈[0.1,10] 对拍")
    max_rel, worst_x = 0.0, 0.0
    for i in range(2001):
        x = 0.1 + (9.9) * i / 2000
        approx = rcp_lookup(x, table)
        truth = 1.0 / x
        rel = abs(approx - truth) / truth
        if rel > max_rel:
            max_rel, worst_x = rel, x
    print(f"    max_rel_err = {max_rel:.2e}  @ x={worst_x:.4f}")

    print("\n小结: 1/x = 查表给初值 + 牛顿迭代收敛。")
    print("  查表解决'初值离真值不远'，迭代解决'精度'（二次收敛，3 轮够）。")
    print("  这是'查表 + 迭代混合'的范式，也是定点除法的地基。")


if __name__ == "__main__":
    main()
