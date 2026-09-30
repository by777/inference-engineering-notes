#!/usr/bin/env python3
"""
step11: ln(x) 查表 + range reduction —— 对数的定点化

和前面所有 step 的关系：
  exp  (step1~5) : exp(x) = 2^n · 2^f       ← 乘法（移位）
  ln   (step11)  : ln(x)  = ln m + e·ln2    ← 加法         ← 本 step
  两者正好是逆运算，所以 step11 还能回头验证前面所有 exp 的误差。

本 step 带来三个前面没出现过的新问题：
  ① range reduction 是【加法】而非乘法 —— 和 exp 对偶
  ② 表域天然是 [1,2)，索引【零乘法】（就是取 float32 尾数的高 8 位）
  ③ 结果【带符号】；而且常数 ln2 的量化误差会被 e 【线性放大】← 本步最大的坑

本 step 展示：
  ① range reduction（加法分解）
  ② 建 ln m 表 [1,2)
  ③ 查表 + 插值 + 加 e·ln2
  ④ 误差分解：哪三项与 x 无关，哪一项随 x 爆炸
  ⑤ 闭环验证 exp(ln(x)) ≈ x
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lut_common import LOG2E, gen_exp2_table, exp2_lookup_lerp

LN2 = math.log(2.0)   # 0.6931471805599453


# ---------------------------------------------------------------------------
# ① range reduction
# ---------------------------------------------------------------------------
def split_mant_exp(x):
    """把 x>0 拆成 m·2^e，m∈[1,2)。

    注意：ln 不需要像 rsqrt 那样把 e 偶化！
    因为 ln 只用到 e·ln2（一个普通乘法），不涉及 e/2 这种半整数移位。
    这是 ln 比 rsqrt 简单的地方。
    """
    m, e = math.frexp(x)      # frexp 给 m∈[0.5,1)
    m = m * 2.0               # 归一化到 [1,2)
    e = e - 1
    return m, e


# ---------------------------------------------------------------------------
# ② 建表
# ---------------------------------------------------------------------------
def gen_ln_table(segs=8, q=12):
    """对 m∈[1,2) 建 ln(m) 表。

    结点 m_k = 1 + k/8，即 1.0, 1.125, ..., 2.0 共 9 个。
    表项 = round(ln(m_k) · 2^q)。

    注意表域是 [1,2) 宽 1 —— 不是 rsqrt 那种 [1,4) 宽 3。
    值域 ln(m) ∈ [0, ln2) = [0, 0.6931)，全为非负，Q12 后 ∈ [0, 2839]。
    """
    width = 1.0                        # 区间 [1,2) 宽 1
    scale = 1 << q                     # 定点标尺 2^q

    table = []
    for k in range(segs + 1):          # 8 段 = 9 个端点
        m_k = 1.0 + k / segs * width   # 第 k 个结点的 m 值
        value = math.log(m_k)          # 该点的函数真值 ln(m_k)
        table.append(round(value * scale))   # 转定点 + 四舍五入
    return table


# ---------------------------------------------------------------------------
# ③ 查表求 ln
# ---------------------------------------------------------------------------
def ln_lookup(x, table, ln2_q=24, q=12):
    """查表 + range reduction 求 ln(x)（x>0）。返回 float 便于对拍。

    ln2_q 是常数 ln2 的定点位数 —— 必须比表项精度 q 高得多，原因见 [4]。
    """
    m, e = split_mant_exp(x)                       # ① x = m·2^e

    # ② 索引：零乘法（见 [3] 的验证）
    #    m∈[1,2) → 8bit；高 3bit 段号、低 5bit 段内位置
    f_byte = min(255, int(round((m - 1.0) * 256)))
    idx = f_byte >> 5
    frac = f_byte & 31

    # ③ 查表 + 两点式插值 → ln(m) 的 Q12 定点值
    lo, hi = table[idx], table[idx + 1]
    ln_m_q12 = (lo * (32 - frac) + hi * frac) >> 5

    # ④ 加上 e·ln2
    #    把 ln(m) 提升到 Q(ln2_q) 再和 e·ln2 相加，避免中间精度损失
    ln2_fixed = round(LN2 * (1 << ln2_q))
    total = ln_m_q12 * (1 << (ln2_q - q)) + e * ln2_fixed

    return total / (1 << ln2_q)


# ---------------------------------------------------------------------------
# ④ 闭环用：一个 exp 查表实现（复用 step1~5 的逻辑）
# ---------------------------------------------------------------------------
def exp_approx(x, exp2_table, q=12):
    """用 exp = 2^(x·log2e) = 2^n·2^f 的查表法算 exp(x)，用于闭环验证。"""
    t = x * LOG2E                      # exp(x) = 2^t
    n = math.floor(t)                  # 整数部分 → 变移位
    f = t - n                          # 小数部分 ∈ [0,1) → 查表

    f_byte = int(round(f * 256))
    if f_byte == 256:                  # 四舍五入后进位到 1.0
        f_byte = 0
        n += 1

    e_q12 = exp2_lookup_lerp(f_byte, exp2_table)   # 2^f 的 Q12 定点值
    if n >= 0:
        out = e_q12 << n               # 乘 2^n = 左移
    else:
        out = e_q12 >> (-n)            # 除 2^n = 右移
    return out / (1 << q)


def main():
    print("=" * 74)
    print("step11  ln(x) 查表 + range reduction（对数的定点化）")
    print("=" * 74)

    # ---------- ① range reduction ----------
    print("\n[1] range reduction: x = m·2^e（m∈[1,2)），则 ln x = ln m + e·ln2")
    print("    对比 exp 的 2^n·2^f（乘法）—— ln 是加法，两者互为逆运算")
    print(f"\n{'x':>11} {'m':>9} {'e':>4} {'e·ln2':>10} {'ln m':>10} "
          f"{'和':>12} {'真值':>12}")
    for x in [1.0, 1.5, 2.0, 3.0, 10.0, 100.0, 1000.0, 0.001]:
        m, e = split_mant_exp(x)
        a = e * LN2
        b = math.log(m)
        print(f"{x:>11} {m:>9.6f} {e:>4} {a:>10.6f} {b:>10.6f} "
              f"{a + b:>12.9f} {math.log(x):>12.9f}")
    print("\n    ★ 注意 e 无需偶化 —— ln 只用 e·ln2（普通乘法），不涉及 e/2")

    # ---------- ② 建表 ----------
    print("\n[2] ln(m) 表 [1,2) 8 段（宽 1，比 rsqrt 的 [1,4) 窄 3 倍）")
    table = gen_ln_table()
    print(f"\n{'k':>2} {'m_k':>8} {'ln m_k':>12} {'×4096':>12} {'round':>8}")
    for k in range(9):
        mk = 1.0 + k / 8
        print(f"{k:>2} {mk:>8.3f} {math.log(mk):>12.8f} "
              f"{math.log(mk) * 4096:>12.4f} {table[k]:>8}")
    print(f"\n    表项 = {table}")
    print(f"    首项 ln(1)=0 → 0（精确）；末项 ln(2)={LN2:.6f} → {table[8]} "
          f"(= {table[8] / 4096:.9f})")

    # ---------- ③ 索引零乘法 ----------
    print("\n[3] 索引零乘法：m∈[1,2) 让 f_byte 就是 float32 尾数的高 8 位")
    print("    A 的 rsqrt 需要 (m-1)/3（除 3，要魔法数乘法）；")
    print("    ln 天然是 [1,2)，索引只需 (m-1)×256 —— 零乘法。")
    import struct
    bad_trunc = 0
    bad_round = 0
    for mant in range(0, 0x800000, 997):
        bits = (127 << 23) | mant                              # 指数=0 → m∈[1,2)
        mv = struct.unpack('<f', struct.pack('<I', bits))[0]    # float32 精确值
        if (mant >> 15) != int((mv - 1.0) * 256):
            bad_trunc += 1
        if ((mant + (1 << 14)) >> 15) != int((mv - 1.0) * 256 + 0.5):
            bad_round += 1
    print(f"    截断 (mant>>15)             == int((m-1)*256)    不符 {bad_trunc}")
    print(f"    圆整 (mant+(1<<14))>>15     == round((m-1)*256)  不符 {bad_round}")
    print("    → 零乘法成立（圆整也只多一条加法）")

    # ---------- ④ 最大的坑：ln2 精度 ----------
    print("\n[4] ⚠️ 最大的坑：ln2 的量化误差会被 e 【线性放大】")
    print("    e·ln2 里的 e 可以很大（float32 的 e 最大约 ±128，float64 约 ±1023）")
    print("    所以 ln2 的定点精度必须远高于表项的 Q12")
    print(f"\n{'ln2 精度':>9} {'ln2 定点值':>12} {'量化误差 Δ':>12} "
          f"{'e=10':>10} {'e=128':>10} {'e=1023':>10}")
    for lq in [12, 16, 20, 24]:
        fixed = round(LN2 * (1 << lq))
        d = abs(fixed / (1 << lq) - LN2)
        print(f"{'Q' + str(lq):>9} {fixed:>12} {d:>12.3e} "
              f"{10 * d:>10.3e} {128 * d:>10.3e} {1023 * d:>10.3e}")
    print("\n    → Q12 时 e=1023 的误差达 3.27e-2，比表本身的误差还大 17 倍！")
    print("    → 若 x 最大到 2^1023（ln x ≈ 709），要用 Q24 才能压到 2e-6")

    # ---------- ⑤ 误差分解 ----------
    print("\n[5] 误差分解：三项与 x 无关，只有 ln2 项随 e 增长")
    h = 1.0 / 8                        # 段宽
    f2_max = 1.0                       # max|f''| = max|-1/m²| = 1 (在 m=1)
    err_interp = h * h / 8 * f2_max    # 线性插值误差上界
    err_index = (1.0 / 256) / 2 * f2_max   # 索引量化：|f'|·step/2，|f'|=1/m 最大 1
    err_table = 0.5 / (1 << 12)        # 表项 Q12 量化：半 LSB
    print(f"    ① 段内线性插值   h²/8·max|f''| = {h}²/8×{f2_max} = {err_interp:.4e}")
    print(f"    ② 索引量化       |f'|·step/2    = 1×{1 / 256:.6f}/2 = {err_index:.4e}")
    print(f"    ③ 表项 Q12 量化  半 LSB        = 0.5/4096      = {err_table:.4e}")
    print(f"    ④ ln2 量化 × |e| Q24           = 1.9e-9×|e|")
    print("    前三项只依赖 m∈[1,2)，与 x 的量级【无关】；")
    print("    第四项随 |e| 线性增长 —— 这就是为什么 ln2 要单独提精度。")
    print(f"\n    本例中 ① ≈ ② 恰好相等（都在 m=1 附近取最大）：")
    print(f"      ① = h²/8·|f''| = (1/8)²/8 × 1 = 1/512 = {err_interp:.4e}")
    print(f"      ② = (h/32)/2·|f'| = (1/256)/2 × 1 = 1/512 = {err_index:.4e}")
    print(f"      原因：ln m 的 |f'|=1/m、|f''|=1/m² 在 m=1 处都等于 1，")
    print(f"            而索引步长恰好是 h/32 = 1/256")
    print(f"      → 这是本配置下的巧合（换 4 段或 16 级就不相等了），")
    print(f"        但它说明【索引量化不能忽略】—— 它和插值误差同量级")

    # ---------- ⑥ 对拍 ----------
    print("\n[6] 全范围对拍（x 跨越 2^-30 ~ 2^30）")
    print(f"\n{'ln2 精度':>9} {'max |绝对误差|':>15} {'max 相对误差':>15} "
          f"{'绝对误差最坏点 x':>20}")
    n = 40000
    lo_x, hi_x = 2.0 ** -30, 2.0 ** 30
    for lq in [12, 24]:
        max_abs = 0.0
        max_rel = 0.0
        worst_x = 0.0
        worst_rel_x = 0.0
        for i in range(n + 1):
            x = lo_x * (hi_x / lo_x) ** (i / n)
            approx = ln_lookup(x, table, ln2_q=lq)
            truth = math.log(x)
            ae = abs(approx - truth)
            if ae > max_abs:
                max_abs = ae
                worst_x = x
            # 相对误差排除 ln x 过零点附近（|ln x| 太小没有意义）
            if abs(truth) > 0.1 and ae / abs(truth) > max_rel:
                max_rel = ae / abs(truth)
                worst_rel_x = x
        print(f"{'Q' + str(lq):>9} {max_abs:>15.4e} {max_rel:>15.4e} "
              f"{worst_x:>20.4e}")
        print(f"{'':>9}   （相对误差最坏点 x = {worst_rel_x:.6f}，"
              f"ln x = {math.log(worst_rel_x):+.4f}）")
    print("\n    ★ 两行的 max 相对误差相同，是因为它出现在 x≈1 附近（|ln x| 小、")
    print("      误差被放大），那里由【插值误差】主导，与 ln2 精度无关。")
    print("      而【绝对误差】最坏点随 ln2 精度明显改善（4.58e-3 → 3.69e-3）。")
    print("\n    ★ ln 会过零（x=1 时 ln x=0），所以【绝对误差】才是有意义的指标；")
    print("      相对误差在 x≈1 附近会失去意义 —— 这与 exp/sigmoid 不同。")

    # ---------- ⑦ 闭环：exp(ln(x)) ----------
    print("\n[7] 闭环验证：exp(ln(x)) ≈ x（把 step1~5 的 exp 拿回来对拍）")
    exp2_table = gen_exp2_table()
    print(f"\n{'x':>12} {'ln 查表':>16} {'exp(ln) 查表':>18} {'相对误差':>14}")
    for x in [0.5, 1.0, 1.5, 2.0, 3.0, 10.0, 100.0]:
        lg = ln_lookup(x, table, ln2_q=24)
        back = exp_approx(lg, exp2_table)
        rel = abs(back - x) / x
        print(f"{x:>12} {lg:>16.9f} {back:>18.9f} {rel:>14.3e}")
    print("\n    → 两个查表流水串联后仍能回到原值（误差 ~1e-3 量级，来自各自的插值误差）")
    print("    → 这说明 exp 和 ln 的 range reduction 是严格互逆的")

    print("\n小结: ln = range reduction（ln m + e·ln2）+ 查表 + 插值。")
    print("  和 exp 对偶（加法 vs 乘法），表域 [1,2) 带来零乘法索引。")
    print("  最大的工程约束不是表，而是【常数 ln2 的精度 × e 的动态范围】。")


if __name__ == "__main__":
    main()
