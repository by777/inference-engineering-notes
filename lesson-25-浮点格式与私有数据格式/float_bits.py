#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lesson-25 浮点格式与私有数据格式 —— 实测脚本

为什么自己写编解码？
    因为 numpy 只给你固定的几种格式。这里要能**任意改 指数位/尾数位/偏置**，
    才能观察到：
      · 位域怎么决定动态范围和精度（anatomy）
      · subnormal 怎么救场、代价是什么（subnormal）
      · 把 fp16 的偏置从 15 改成别的值 = 私有格式 fp16e/fp16s 的原理（scale）
      · 加宽内部容器怎么避免 fp16 累加的"吸收"（accumulate）

子命令：
    python3 float_bits.py anatomy       # 拆解位域，验证解码公式
    python3 float_bits.py subnormal     # subnormal 边界与精度线性退化 + FTZ
    python3 float_bits.py scale         # 量化 scale 在 fp16 / fp16e 下的精度
    python3 float_bits.py accumulate    # fp16 累加 vs 加宽容器 vs 精确值
    python3 float_bits.py all

只依赖标准库。
"""
from __future__ import annotations

import math
import random
import sys

# --------------------------------------------------------------------------
# 一、通用 IEEE754 风格编解码
#    位布局：[sign(1)] [exp(eb)] [mantissa(mb)]
#    值   = (-1)^s * 2^(E-bias) * (1 + M/2^mb)          E in [1, 2^eb-2]  正规数
#         = (-1)^s * 2^(1-bias)  * (0 + M/2^mb)          E = 0             subnormal
#         = ±inf / NaN                                   E = 2^eb-1
# --------------------------------------------------------------------------

FORMATS: dict[str, tuple[int, int, int]] = {
    #  名称          (exp_bits, man_bits, bias)
    "fp64":         (11, 52, 1023),
    "fp32":         (8, 23, 127),
    "tf32":         (8, 10, 127),   # NVIDIA TensorFloat-32：fp32 指数 + fp16 尾数
    "bf16":         (8, 7, 127),
    "fp16":         (5, 10, 15),
    "fp8_e4m3":     (4, 3, 7),
    "fp8_e5m2":     (5, 2, 15),
}


def encode(x: float, eb: int, mb: int, bias: int, subnormal: bool = True) -> int:
    """把实数编码成位模式（int）。subnormal=False 模拟 FTZ（flush-to-zero）。"""
    sign = 1 if math.copysign(1.0, x) < 0.0 else 0
    a = abs(float(x))
    emax_field = (1 << eb) - 1
    inf_pat = (sign << (eb + mb)) | (emax_field << mb)
    if math.isnan(a):
        return inf_pat | 1
    if math.isinf(a):
        return inf_pat
    if a == 0.0:
        return sign << (eb + mb)

    _m, e = math.frexp(a)                 # a = m * 2^e, 0.5 <= m < 1
    E = e - 1 + bias                      # 正规数情况下的指数字段
    if E >= 1:
        frac = a / math.ldexp(1.0, e - 1) - 1.0        # a = (1+frac) * 2^(e-1)
        M = int(round(frac * (1 << mb)))               # round() 是 half-even
        if M >= (1 << mb):                             # 尾数进位 -> 指数 +1
            M = 0
            E += 1
        if E >= emax_field:
            return inf_pat
        return (sign << (eb + mb)) | (E << mb) | M

    # a 落在 subnormal 区（E 本来会 <= 0）
    if not subnormal:
        return sign << (eb + mb)                       # FTZ：直接刷成 0
    ulp = math.ldexp(1.0, 1 - bias - mb)               # subnormal 的固定间距
    M = int(round(a / ulp))
    if M >= (1 << mb):                                 # 圆整后进位到最小正规数
        return (sign << (eb + mb)) | (1 << mb)
    return (sign << (eb + mb)) | M


def decode(bits: int, eb: int, mb: int, bias: int) -> float:
    """位模式（int）-> 实数。"""
    sign = -1.0 if (bits >> (eb + mb)) & 1 else 1.0
    E = (bits >> mb) & ((1 << eb) - 1)
    M = bits & ((1 << mb) - 1)
    if E == (1 << eb) - 1:                             # 全 1：inf / NaN
        return float("nan") if M else sign * float("inf")
    if E == 0:                                         # subnormal / 零
        return sign * math.ldexp(M, 1 - bias - mb)
    return sign * math.ldexp(1.0 + M / (1 << mb), E - bias)


def to_format(x: float, eb: int, mb: int, bias: int, subnormal: bool = True) -> float:
    """往返一次：实数 -> 该格式 -> 实数。"""
    return decode(encode(x, eb, mb, bias, subnormal), eb, mb, bias)


def bits_str(bits: int, eb: int, mb: int) -> str:
    total = 1 + eb + mb
    s = format(bits, f"0{total}b")
    return f"{s[0]} {s[1:1 + eb]} {s[1 + eb:]}"


def fmt(x: float) -> str:
    if x == 0.0:
        return "0"
    if math.isinf(x):
        return "inf"
    if math.isnan(x):
        return "nan"
    a = abs(x)
    if a < 1e-4 or a >= 1e5:
        return f"{x:.4e}"
    return f"{x:.6g}"


# --------------------------------------------------------------------------
# 二、加宽内部容器（fpin 模拟）
#    把任意 double 舍入到 N 个有效二进制位。N = 1(符号隐含) + 24 尾数 = 25。
# --------------------------------------------------------------------------

def round_sig(x: float, nbits: int) -> float:
    """舍入到 nbits 个有效二进制位（half-even），模拟有限精度内部累加器。"""
    if x == 0.0 or not math.isfinite(x):
        return x
    m, e = math.frexp(x)                  # 0.5 <= |m| < 1
    scaled = m * (1 << nbits)             # 落在 [2^(nbits-1), 2^nbits)
    r = round(scaled)                     # half-even
    if abs(r) >= (1 << nbits):
        r /= 2.0
        e += 1
    return math.ldexp(r / (1 << nbits), e)


# --------------------------------------------------------------------------
# 三、各子命令
# --------------------------------------------------------------------------

def cmd_anatomy() -> None:
    eb16, mb16, b16 = FORMATS["fp16"]
    print("=" * 78)
    print("【1】位域解剖：指数位买动态范围，尾数位买精度")
    print("=" * 78)
    head = f"{'格式':<10}{'位域':<10}{'偏置':>5}{'有效位':>7}{'最小正规数':>13}{'最大正规数':>13}{'数量级跨度':>11}"
    print(head)
    print("-" * 78)
    for name, (eb, mb, bias) in FORMATS.items():
        lo = to_format(math.ldexp(1.0, 1 - bias), eb, mb, bias)          # E=1,M=0
        hi = to_format(math.ldexp((1.0 + ((1 << mb) - 1) / (1 << mb)), (1 << eb) - 2 - bias), eb, mb, bias)
        span = math.log10(hi) - math.log10(lo)
        layout = f"1+{eb}+{mb}"
        note = ""
        if name == "fp8_e4m3":
            note = "  注：本脚本是通用 IEEE 模型（全1指数留给 inf/NaN），真实 E4M3 无 inf、最大 448"
        print(f"{name:<10}{layout:<10}{bias:>5}{mb + 1:>7}{fmt(lo):>13}{fmt(hi):>13}{span:>10.2f}x{note}")

    print()
    print("--- 同一个数 1.5 在各格式里的位模式与往返误差 ---")
    x = 1.5
    for name, (eb, mb, bias) in FORMATS.items():
        b = encode(x, eb, mb, bias)
        y = decode(b, eb, mb, bias)
        print(f"{name:<10} {bits_str(b, eb, mb)}  ->  {fmt(y)}   误差 {abs(y - x):.3e}")

    print()
    print("--- 关键门槛（fp16）：解码公式逐项核对 ---")
    mn = to_format(math.ldexp(1.0, 1 - b16), eb16, mb16, b16)
    ulp_sub = math.ldexp(1.0, 1 - b16 - mb16)
    print(f"  最小正规数        = 2^(1-{b16})        = 2^-14       = {fmt(mn)}")
    print(f"  subnormal 间距    = 2^(1-{b16}-{mb16})   = 2^-24       = {fmt(ulp_sub)}")
    print(f"  subnormal 个数    = 2^{mb16}-1          = {(1 << mb16) - 1}")
    print(f"  最大正规数        = 2^-14 * (2-2^-10)  = {fmt(to_format(65504.0, eb16, mb16, b16))}")
    print(f"  溢出阈值          = 65504；超过 -> inf，实测 65520 -> {fmt(to_format(65520.0, eb16, mb16, b16))}")


def cmd_subnormal() -> None:
    eb, mb, bias = FORMATS["fp16"]
    eb32, mb32, b32 = FORMATS["fp32"]
    eb_bf, mb_bf, b_bf = FORMATS["bf16"]

    print("=" * 78)
    print("【2】subnormal：指数用光了，借尾数位继续往下表示")
    print("=" * 78)

    print("--- 边界：位模式只差最后一位，值却跨过 2^-14 ---")
    max_sub = decode((1 << mb) - 1, eb, mb, bias)          # E=0, M=全1
    min_norm = decode(1 << mb, eb, mb, bias)               # E=1, M=0
    print(f"  最小正规数     {bits_str(1 << mb, eb, mb)}  -> {fmt(min_norm)}")
    print(f"  最大 subnormal {bits_str((1 << mb) - 1, eb, mb)}  -> {fmt(max_sub)}")
    ulp = math.ldexp(1.0, 1 - bias - mb)
    print(f"  最大 subnormal + ulp == 最小正规数 ? {max_sub + ulp == min_norm}")
    print(f"  -> 间距固定 = {fmt(ulp)} = 2^-24，无缝衔接（gradual underflow）")

    print()
    print("--- 采样：subnormal 区里 exp 字段恒为 0 ---")
    for v in (2.0 ** -16, 2.0 ** -20, 2.0 ** -24):
        b = encode(v, eb, mb, bias)
        print(f"  {fmt(v):>12} -> {bits_str(b, eb, mb)}  (exp=00000 => subnormal)")

    print()
    print("--- subnormal 的代价：有效位数线性退化（值 = M * 2^-24）---")
    print(f"  {'M':>6}{'需要的位数':>11}{'值':>15}{'相对精度':>11}")
    for M in (1023, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
        val = decode(M, eb, mb, bias)
        print(f"  {M:>6}{M.bit_length():>11}{fmt(val):>15}{M.bit_length():>10}bit")
    print("  -> 指数不能再减了，只好牺牲尾数位：这叫用精度换连续性")

    print()
    print("--- 三种格式都有 subnormal 机制（个数 = 2^尾数位 - 1）---")
    print(f"  {'格式':<8}{'偏置':>6}{'尾数位':>8}{'最小正规数':>15}{'subnormal 间距':>17}{'subnormal 个数':>15}")
    for name, (e_, m_, b_) in (("fp32", (eb32, mb32, b32)), ("bf16", (eb_bf, mb_bf, b_bf)), ("fp16", (eb, mb, bias))):
        lo = math.ldexp(1.0, 1 - b_)
        u = math.ldexp(1.0, 1 - b_ - m_)
        print(f"  {name:<8}{b_:>6}{m_:>8}{fmt(lo):>15}{fmt(u):>17}{(1 << m_) - 1:>15}")

    print()
    print("--- FTZ 风险：不支持 subnormal 的硬件会把它们刷成 0 ---")
    for v in (1e-5, 1e-7, 1e-8):
        ieee = to_format(v, eb, mb, bias, subnormal=True)
        ftz = to_format(v, eb, mb, bias, subnormal=False)
        print(f"  输入 {v:.0e}  IEEE(有subnormal) -> {fmt(ieee):>12}    FTZ(刷0) -> {fmt(ftz):>12}")
    print("  -> 1e-7 在 IEEE 下是一个 subnormal，还能用；在 FTZ 硬件上直接变 0")
    print("  -> 量化 scale 一旦归零，整层输出全废：这是模拟器与板子不一致的经典来源")


def cmd_scale() -> None:
    eb, mb, bias = FORMATS["fp16"]
    print("=" * 78)
    print("【3】量化 scale 的表示域困境 —— 私有格式 fp16s/fp16e 的由来")
    print("=" * 78)
    print("scale = max|w| / 127，权重越大 scale 越大；小模型的 scale 会很小。")
    print()

    print("--- 标准 fp16（bias 固定 15）：scale 越小，有效位掉得越狠 ---")
    print(f"  {'max|w|':>9}{'scale':>13}{'理想指数':>10}{'落点':>12}{'实存值':>14}{'相对误差':>11}{'有效位':>8}")
    for mx in (0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6):
        s = mx / 127
        e_exp = math.floor(math.log2(s))          # s 约等于 2^e_exp
        E = e_exp + bias
        stored = to_format(s, eb, mb, bias)
        rel = abs(stored - s) / s
        if E >= 1:
            seat, sig = "正规数", mb + 1
        else:
            M = round(s / math.ldexp(1.0, 1 - bias - mb))
            seat, sig = "subnormal", max(1, M.bit_length())
        print(f"  {mx:>9.0e}{s:>13.4e}{E:>10}{seat:>12}{stored:>14.6e}{rel * 100:>10.2f}%{sig:>7}bit")

    print()
    print("--- 私有格式 fp16e：把偏置 e0 变成可配参数（re-bias / 滑动窗口）---")
    print("    原理一句话：e0 把该张量的量级搬到 1 附近，窗口就跟着张量走。")
    s = 1e-6
    e_exp = math.floor(math.log2(s))
    print(f"    目标 scale = {fmt(s)}，其二进制量级 = 2^{e_exp}")
    print()
    print(f"  {'e0':>5}{'scale 的 E':>12}{'合法?':>7}{'窗口下界':>14}{'窗口上界':>13}{'有效位':>8}")
    for e0 in (15, 23, 31, 35, 45):
        E = e_exp + e0
        lo = math.ldexp(1.0, 1 - e0)
        hi = math.ldexp(1.0 + ((1 << mb) - 1) / (1 << mb), (1 << eb) - 2 - e0)
        ok = "✓" if 1 <= E <= (1 << eb) - 2 else "✗"
        sig = f"{mb + 1}bit" if ok == "✓" else "—"
        mark = "   <- 量级归一化到这里（E=15）" if e0 == 35 else ("   <- 本文前面用的值" if e0 == 31 else "")
        print(f"  {e0:>5}{E:>12}{ok:>7}{fmt(lo):>14}{fmt(hi):>13}{sig:>8}{mark}")
    print()
    print("    注意：窗口宽度永远是 2^30 ≈ 9.03 个数量级（指数位固定 5 位），")
    print("         e0 只决定这个窗口摆在哪一段实数轴上。")
    print()

    e0 = 35
    stored_fp16 = to_format(s, eb, mb, bias)
    stored_fp16e = to_format(s, eb, mb, e0)
    print("--- 同一个 16 位容器，只改偏置，精度差多少 ---")
    print(f"  fp16  (e0=15) : {stored_fp16:.6e}   相对误差 {abs(stored_fp16 - s) / s * 100:6.2f}%")
    print(f"  fp16e (e0=35) : {stored_fp16e:.6e}   相对误差 {abs(stored_fp16e - s) / s * 100:6.2f}%")
    gain = math.log2((abs(stored_fp16 - s) / s) / (abs(stored_fp16e - s) / s))
    print(f"  精度提升约 {gain:.1f} bit —— 这就是 fp16s / fp16e 存在的理由")
    print()
    print("  ⚠️ 前提：re-bias 只能把 subnormal 升级成正规数，")
    print("     对本来就在正规区的张量，一点精度都不多给。")


def cmd_accumulate() -> None:
    eb, mb, bias = FORMATS["fp16"]
    print("=" * 78)
    print("【4】为什么芯片需要更宽的内部容器（fpin 33bit 的动机）")
    print("=" * 78)

    print("--- 4.1 吸收阈值：1.0 上面加一个微小量，能不能改变结果 ---")
    print("    （ulp 是'1.0 处的最小可分辨增量'；比半个 ulp 还小的加数会被直接丢掉）")
    scan: dict[int, tuple[bool, bool]] = {}
    for p in range(1, 28):
        t = math.ldexp(1.0, -p)
        scan[p] = (to_format(1.0 + t, eb, mb, bias) != 1.0, round_sig(1.0 + t, 25) != 1.0)
    # 加数越大越容易生效；阈值 = 最后一个仍能生效的阶
    last16 = max(p for p in sorted(scan) if scan[p][0])
    lastwide = max(p for p in sorted(scan) if scan[p][1])
    print(f"  {'阶':>4}{'加数 2^-p':>14}{'fp16 生效?':>12}{'加宽容器(25bit) 生效?':>24}")
    for p in range(9, 28):
        ok16, okw = scan[p]
        if p == last16:
            seam = "   <- fp16 的阈值"
        elif p == lastwide:
            seam = "   <- 加宽容器的阈值"
        else:
            seam = ""
        print(f"  {p:>4}{fmt(math.ldexp(1.0, -p)):>14}{str(ok16):>12}{str(okw):>24}{seam}")
    print(f"  -> fp16 要求加数 >= 2^-{last16}；加宽容器要求 >= 2^-{lastwide}")
    print(f"  -> 抗吸收余量放大 2^{lastwide - last16} = {2 ** (lastwide - last16)} 倍")
    print("  -> 等价说法：1.0 处的最小可分辨增量从 fp16 的 2^-10 压到容器内的 2^-24")

    print()
    print("--- 4.2 长累加：加 K 个 1.0，看结果对不对 ---")
    print(f"  {'K':>8}{'fp16 顺序加':>16}{'加宽容器(25bit)':>18}{'精确值':>12}{'fp16 误差':>12}")
    for K in (1000, 10000, 70000):
        s16 = 0.0
        for _ in range(K):
            s16 = to_format(s16 + 1.0, eb, mb, bias)
        sw = 0.0
        for _ in range(K):
            sw = round_sig(sw + 1.0, 25)
        err = abs(s16 - K) / K * 100 if math.isfinite(s16) else float("nan")
        print(f"  {K:>8}{fmt(s16):>16}{fmt(sw):>18}{K:>12}{err:>11.2f}%")
    print("  -> fp16 在 2048 附近就停住了：再加 1.0 只是半个 ulp，被 round-half-even 丢掉")
    print("  -> 这不是溢出（fp16 上限 65504），而是**吸收**：大数与小数相加，小数消失了")

    print()
    print("--- 4.3 与精确值对比：更真实的点积（K 项，量级跨 6 个数量级）---")
    print(f"  {'K':>6}{'fp16 累加偏差':>16}{'加宽容器偏差':>16}{'改善倍数':>12}")
    for K in (64, 256, 1024, 4096):
        terms = make_terms(K)
        exact = math.fsum(terms)
        sf = 0.0
        for t in terms:
            sf = to_format(sf + t, eb, mb, bias)
        sw = 0.0
        for t in terms:
            sw = round_sig(sw + t, 25)
        ref = abs(exact)
        ef = abs(sf - exact) / ref if math.isfinite(sf) else float("inf")
        ew = abs(sw - exact) / ref if math.isfinite(sw) else float("inf")
        ratio = ef / ew if ew else float("inf")
        print(f"  {K:>6}{ef:>15.3e}{ew:>16.3e}{ratio:>11.1f}x")
    print()
    print("  注：加宽容器仍非零误差——指数差超过 25 位时一样会舍入，")
    print("     只是阈值抬高、且'每步都舍'变成'少数几步才舍'。")


def make_terms(K: int, seed: int = 20260921) -> list[float]:
    """生成 K 个 fp16 项，量级跨 6 个数量级（模拟真实激活/权重的长尾）。"""
    rnd = random.Random(seed)
    eb, mb, bias = FORMATS["fp16"]
    out = []
    for _ in range(K):
        mag = 10.0 ** rnd.uniform(-3.0, 3.0)
        v = mag * (1.0 if rnd.random() < 0.5 else -1.0) * rnd.uniform(0.5, 1.5)
        out.append(to_format(v, eb, mb, bias))
    return out


CMDS = {
    "anatomy": cmd_anatomy,
    "subnormal": cmd_subnormal,
    "scale": cmd_scale,
    "accumulate": cmd_accumulate,
}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in (*CMDS, "all"):
        print(__doc__)
        return 0
    names = list(CMDS) if argv[1] == "all" else [argv[1]]
    for i, n in enumerate(names):
        if i:
            print()
        CMDS[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
