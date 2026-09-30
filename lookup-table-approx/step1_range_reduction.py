#!/usr/bin/env python3
"""
step1: range reduction —— 为什么 exp 能拆成"移位 + 查 [0,1)"？

核心恒等式：
    exp(x) = 2^(x·log2e)
           = 2^n · 2^f
    其中 n = floor(x·log2e) 为整数，f = x·log2e - n ∈ [0,1)

意义：
    · 整数 n 部分：2^n 在定点/二进制里只是一个"移位"，零成本
    · 小数 f 部分：永远落在 [0,1)，只需覆盖这一个单位区间 —— 表可以很小
    · softmax 里 x 是 (logit - max)，恒 ≤ 0，所以 n ≤ 0，全是右移
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lut_common import LOG2E

def split_pow2(t: float):
    """把任意实数 t 拆成 (n, f)，满足 2^t = 2^n · 2^f, f∈[0,1)"""
    n = math.floor(t)
    f = t - n
    return n, f


def main():
    print("=" * 72)
    print("step1  range reduction: exp(x) = 2^n · 2^f")
    print("=" * 72)

    # ---------- 1. 恒等式验证 ----------
    print("\n[1] 恒等式验证: 随机 x，exp(x) 与 2^n·2^f 是否一致")
    print(f"{'x':>10} {'x·log2e':>10} {'n':>5} {'f':>8}  {'exp(x)':>12} {'2^n·2^f':>12}  match")
    for x in [0.0, 0.5, -0.5, 3.0, -3.0, 7.3, -10.0]:
        t = x * LOG2E
        n, f = split_pow2(t)
        lhs = math.exp(x)
        rhs = (2.0 ** n) * (2.0 ** f)      # 数学等价，2^n 在硬件里是移位
        ok = math.isclose(lhs, rhs, rel_tol=1e-12)
        print(f"{x:>10.3f} {t:>10.4f} {n:>5d} {f:>8.4f}  {lhs:>12.6f} {rhs:>12.6f}  {ok}")

    # ---------- 2. 为什么只查 [0,1) ----------
    print("\n[2] 关键观察: f 永远 ∈ [0,1)，表只需覆盖一个单位区间")
    print("    n 只负责'移位': n=1 → ×2, n=-3 → ÷8 (右移3位)")
    print("    所以真正的'函数求值'只剩 2^f, f∈[0,1) 这一段。")

    # ---------- 3. 延伸到 softmax（x = logit - max ≤ 0） ----------
    print("\n[3] softmax 语境: x = logit - max ≤ 0 → n ≤ 0 → 全部右移")
    logits = [3.2, 1.5, 0.0, -2.0, -5.0]
    xmax = max(logits)
    print(f"    logits = {logits}, max = {xmax}")
    print(f"{'logit':>7} {'x=logit-max':>12} {'t=x·log2e':>11} {'n':>4} {'f':>8}")
    for z in logits:
        x = z - xmax
        t = x * LOG2E
        n, f = split_pow2(t)
        print(f"{z:>7.1f} {x:>12.3f} {t:>11.4f} {n:>4d} {f:>8.4f}")

    # ---------- 4. 若不用 range reduction 会怎样 ----------
    print("\n[4] 为什么非做不可（不做会怎样）")
    print("    直接对 exp 建表: 输入动态范围可以很大（logit 差几十），")
    print("    若均匀分段覆盖 [-20, 0]，段宽 h=20/8=2.5，线性化误差 ~ h²/8·max|f''|")
    err_no_rr = (20/8)**2 / 8 * math.exp(0)   # 2^x 换回 exp 粗略估
    err_with_rr = (1/8)**2 / 8 * (math.log(2)**2) * 2.0  # 只查 [0,1)，x→1 端最大
    print(f"    不压缩(覆盖[-20,0],8段):   线性化误差 ≈ {err_no_rr:.4f}")
    print(f"    压缩后(只查[0,1),8段):     线性化误差 ≈ {err_with_rr:.5f}")
    print(f"    → 相差约 {err_no_rr/err_with_rr:.0f} 倍。范围压缩是表能建小的前提。")

    print("\n小结: exp(x) = 2^n·2^f；n→移位（零成本），f∈[0,1)→只需要 1 个单位区间的小表。")
    print("下一步 step2 看这张 9 项的表具体是怎么算出来的。")


if __name__ == "__main__":
    main()
