#!/usr/bin/env python3
"""
step12: 非均匀分段 — 什么时候值得，什么时候不值得

前面的 step 全部用【均匀分段】。step5 结尾提过"非均匀分段"这个优化方向，
step6 也说过 sigmoid 的饱和区可以粗分。本 step 把它讲清楚，并给出一个
反直觉的结论。

本 step 展示：
  ① 曲率变化比 —— 决定"非均匀潜力"的唯一指标
  ② 均匀分段的逐段误差不均衡（ln 的实测）
  ③ 等误差最优网格推导 → 几何网格 2^(k/8)
  ④ ⚠️ 核心洞察：非均匀撞上【编码冲突】，在 3bit+5bit 编码下结构上不可能
  ⑤ 公平对比：加表项 vs 非均匀分段（结论：加表项完胜）
  ⑥ 误差配平：为什么"只加段数"会撞地板
  ⑦ 非均匀真正值得的场景（饱和型函数）
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

Q = 12          # 表项定点位数
FRAC_LEVELS = 32   # 默认段内电平数（5bit）


# ---------------------------------------------------------------------------
# 公共：建表 + 两种索引
# ---------------------------------------------------------------------------
def gen_table(breaks, q=Q):
    """给定段边界列表，生成表项 table[k] = round(ln(breaks[k])·2^q)。"""
    scale = 1 << q
    table = []
    for b in breaks:
        table.append(round(math.log(b) * scale))
    return table


def make_uniform_breaks(segs):
    """均匀分段边界：1 + k/segs，k=0..segs。"""
    breaks = []
    for k in range(segs + 1):
        breaks.append(1.0 + k / segs)
    return breaks


def make_geo_breaks(segs):
    """几何分段边界：2^(k/segs)，k=0..segs（等误差最优网格）。"""
    breaks = []
    for k in range(segs + 1):
        breaks.append(2.0 ** (k / segs))
    return breaks


def lookup_uniform(m, table, seg_bits, frac_bits):
    """均匀字节索引：f_byte = (m-1)·2^(seg+frac)，高位段号、低位 frac。

    这是硬件最爱的形式 —— 段号和 frac 靠【一条移位 + 一条掩码】拿到，
    零乘法、零比较。
    """
    segs = 1 << seg_bits
    levels = 1 << (seg_bits + frac_bits)
    fb = int(round((m - 1.0) * levels))
    if fb >= levels:               # m→2 时向上取整溢出，clamp 到最后电平
        fb = levels - 1
    idx = fb >> frac_bits
    frac = fb & ((1 << frac_bits) - 1)
    n = 1 << frac_bits
    lo, hi = table[idx], table[idx + 1]
    return ((lo * (n - frac) + hi * frac) >> frac_bits) / (1 << Q)


def lookup_geo(m, table, segs):
    """几何网格索引：需要 t = segs·log2(m) —— 这就是代价。

    本函数用浮点 log2 只是为了【测量分段质量】，
    真实硬件上这一步要么另建 log2 表、要么用比较树/二分。
    """
    t = math.log2(m) * segs
    fb = int(round(t * FRAC_LEVELS))
    if fb > segs * FRAC_LEVELS - 1:
        fb = segs * FRAC_LEVELS - 1
    idx = fb >> 5
    frac = fb & 31
    lo, hi = table[idx], table[idx + 1]
    return ((lo * (32 - frac) + hi * frac) >> 5) / (1 << Q)


def max_err_uniform(segs, seg_bits, frac_bits, n=20000):
    """扫描均匀方案，返回 (全域max, 全域最坏点, 内点max, 内点最坏点, 表项数)。

    内点取 m ≤ 1.996（避开顶端 1~2 个索引电平的端点截断效应）。
    """
    table = gen_table(make_uniform_breaks(segs))
    mf = 0.0
    wf = 0.0
    mi = 0.0
    wi = 0.0
    for i in range(n + 1):
        m = 1.0 + (1.0 - 1e-12) * i / n
        e = abs(lookup_uniform(m, table, seg_bits, frac_bits) - math.log(m))
        if e > mf:
            mf = e
            wf = m
        if m <= 1.996 and e > mi:
            mi = e
            wi = m
    return mf, wf, mi, wi, len(table)


def max_err_geo(segs, n=20000):
    """几何网格方案，返回格式同 max_err_uniform。"""
    table = gen_table(make_geo_breaks(segs))
    mf = 0.0
    wf = 0.0
    mi = 0.0
    wi = 0.0
    for i in range(n + 1):
        m = 1.0 + (1.0 - 1e-12) * i / n
        e = abs(lookup_geo(m, table, segs) - math.log(m))
        if e > mf:
            mf = e
            wf = m
        if m <= 1.996 and e > mi:
            mi = e
            wi = m
    return mf, wf, mi, wi, len(table)


def main():
    print("=" * 74)
    print("step12  非均匀分段 — 什么时候值得")
    print("=" * 74)

    # ------------------------------------------------------------------
    # ① 曲率变化比
    # ------------------------------------------------------------------
    print("\n[1] 曲率变化比 = max|f''| / min|f''|（在查表区间内）")
    print("    这是决定'非均匀潜力'的唯一指标：变化比大 → 均匀分段浪费大")
    print()

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    # 注意：从 i=1 开始采样 —— sigmoid/tanh 的 f''(0)=0 是【拐点】，
    # 不是"平坦"，放进 min 会让比值虚高
    def ratio(f2, a, b, n=200000):
        mx = 0.0
        mn = float("inf")
        for i in range(1, n + 1):
            x = a + (b - a) * i / n
            v = abs(f2(x))
            if v > mx:
                mx = v
            if v < mn:
                mn = v
        return mx, mn

    cases = []
    cases.append(("exp 的 2^f, f∈[0,1)",
                  lambda x: math.log(2) ** 2 * 2.0 ** x, 0.0, 1.0))
    cases.append(("ln m, m∈[1,2)", lambda x: 1.0 / (x * x), 1.0, 2.0))
    cases.append(("rsqrt 的 1/sqrt(m), m∈[1,4)",
                  lambda x: 0.75 * x ** -2.5, 1.0, 4.0))
    cases.append(("sigmoid, x∈[0,8]（含饱和区）",
                  lambda x: sigmoid(x) * (1 - sigmoid(x)) * (1 - 2 * sigmoid(x)),
                  0.0, 8.0))
    cases.append(("tanh, x∈[0,4]（含饱和区）",
                  lambda x: -2 * math.tanh(x) / math.cosh(x) ** 2, 0.0, 4.0))

    print(f"{'函数 / 区间':>32} {'max|f2|':>11} {'min|f2|':>11} {'变化比':>9}")
    print("-" * 74)
    for name, f2, a, b in cases:
        mx, mn = ratio(f2, a, b)
        print(f"{name:>32} {mx:>11.4e} {mn:>11.4e} {mx / mn:>9.0f}")
    print()
    print("    读法：exp 只差 2 倍（几乎均匀）→ 非均匀没意义；")
    print("          ln 差 4 倍 → 有空间但不大；")
    print("          sigmoid/tanh 差【上万倍】→ 饱和区极平坦，非均匀收益大")

    # ------------------------------------------------------------------
    # ② 均匀分段的逐段误差
    # ------------------------------------------------------------------
    print("\n[2] 均匀 8 段的逐段误差（ln）：看是否均衡")
    h = 1.0 / 8
    table8 = gen_table(make_uniform_breaks(8))
    print()
    print(f"{'段':>3} {'m 范围':>15} {'max|f2|':>9} {'插值上界':>10} "
          f"{'实测':>10} {'索引量化':>10}")
    for k in range(8):
        lo_m = 1.0 + k * h
        hi_m = 1.0 + (k + 1) * h
        f2 = 1.0 / (lo_m * lo_m)
        bound = h * h / 8 * f2
        mx = 0.0
        n = 600
        for i in range(n + 1):
            m = lo_m + (hi_m - lo_m) * i / n
            if m >= 2.0:
                m = math.nextafter(2.0, 0)
            e = abs(lookup_uniform(m, table8, 3, 5) - math.log(m))
            if e > mx:
                mx = e
        idx_err = (1.0 / lo_m) * (1.0 / 256) / 2
        print(f"{k:>3} [{lo_m:.4f},{hi_m:.4f}) {f2:>9.4f} {bound:>10.2e} "
              f"{mx:>10.2e} {idx_err:>10.2e}")
    print()
    print("    → 段 0 误差 ~3.8e-3，段 6 只有 ~1.8e-3，差约 2 倍（不均衡）")
    print("    → 但注意：段尾（k=7）误差反而回升，那是 m→2 的【端点截断】效应")

    # ------------------------------------------------------------------
    # ③ 等误差最优网格
    # ------------------------------------------------------------------
    print("\n[3] 等误差最优网格推导：对 ln 得到【几何网格】")
    print()
    print("    第 k 段宽 h_k、左端 m_k。ln 的 |f''| = 1/m² 递减，")
    print("    故段内 max|f''| = 1/m_k²，插值误差 = h_k²/(8·m_k²)")
    print("    要各段相等 → h_k²/m_k² = 常数 → h_k ∝ m_k")
    print("    而 m_{k+1} = m_k + h_k = m_k(1+K) → 几何级数")
    print("    端点 m_8 = (1+K)^8 = 2 → 1+K = 2^(1/8)")
    print("    ∴ 最优边界 m_k = 2^(k/8)")
    print()
    K = 2.0 ** (1 / 8) - 1
    print(f"    K = 2^(1/8) - 1 = {K:.6f}，各段 h_k/m_k 恒等于 K")
    print()
    print(f"{'k':>2} {'边界 2^(k/8)':>13} {'段宽 h_k':>11} {'h_k/最窄':>9}")
    h0 = K * 1.0
    for k in range(9):
        mk = 2.0 ** (k / 8)
        hk = K * mk if k < 8 else 0.0
        hh = hk / h0 if k < 8 else 0.0
        print(f"{k:>2} {mk:>13.6f} {hk:>11.6f} {hh:>9.3f}")
    print()
    print(f"    ⚠️ 最宽段/最窄段 = {2.0 ** (7 / 8):.3f} 倍 —— 只有 {2.0 ** (7 / 8):.2f} 倍！")
    print("    最优非均匀只是【轻微】不均匀，所以可挖空间本来就小")

    # ------------------------------------------------------------------
    # ④ 编码冲突（核心洞察）
    # ------------------------------------------------------------------
    print("\n[4] ⚠️ 核心洞察：非均匀撞上【编码冲突】")
    print()
    min_w = 32.0 / 256.0
    print(f"    标准编码：f_byte = (m-1)×256，取 8bit")
    print(f"      高 3bit = 段号（8 段），低 5bit = frac（32 个电平）")
    print()
    print(f"    8 段 × 32 电平 = 256 电平，而 m∈[1,2) 只有 256 个字节可放")
    print(f"    → 每段【恰好】占 32 个字节 → 每段宽【必须】= 32/256 = {min_w:.6f}")
    print(f"    → 也就是【只能均匀分段】！")
    print()
    print(f"    再看几何最优的最窄段：h_0 = {K:.6f}")
    print(f"    判定: {K:.6f} < {min_w:.6f}  →  ❌ 装不下")
    print()
    print("    结论：在「3bit 段号 + 5bit frac + 均匀字节编码」下，")
    print("          非均匀分段【结构上不可能】。")
    print("    要非均匀，必须换索引机制：")
    print("      · 加索引位（≥9bit）+ 非线性映射")
    print("      · 比较树 / 二分查找定位段号")
    print("      · 预存一张「字节→段号」的小 LUT")
    print("    而每一种都意味着【额外逻辑】，不是免费午餐。")

    # ------------------------------------------------------------------
    # ⑤ 公平对比
    # ------------------------------------------------------------------
    print("\n[5] 公平对比：非均匀 vs 加表项 vs 加索引位")
    print()
    print("    内点 = m≤1.996（避开顶端电平截断）；两者差别见 [6]")
    print()
    print(f"{'方案':>20} {'位':>3} {'表项':>5} {'全域max':>10} {'内点max':>10} "
          f"{'索引成本':>10}")
    print("-" * 74)
    rows = []
    rows.append(("均匀 8段×32级", 8, 3, 5, "零"))
    rows.append(("均匀 16段×16级", 8, 4, 4, "零"))
    rows.append(("均匀 32段×8级", 8, 5, 3, "零"))
    rows.append(("均匀 16段×32级", 9, 4, 5, "零"))
    rows.append(("均匀 32段×32级", 10, 5, 5, "零"))
    for name, bits, sb, fb, cost in rows:
        mf, wf, mi, wi, ntab = max_err_uniform(1 << sb, sb, fb)
        print(f"{name:>20} {bits:>3} {ntab:>5} {mf:>10.3e} {mi:>10.3e} "
              f"{cost:>10}")
    for name, segs, bits in [("几何 8段×32级", 8, "~9"),
                             ("几何 16段×32级", 16, "~10")]:
        mf, wf, mi, wi, ntab = max_err_geo(segs)
        print(f"{name:>20} {bits:>3} {ntab:>5} {mf:>10.3e} {mi:>10.3e} "
              f"{'需 log2':>10}")
    print()
    print("    三条关键对比：")
    print("    ① 同为 8bit/9 项：")
    print("         均匀 8段×32级  内点 3.82e-3")
    print("         几何 8段×32级  内点 1.68e-3  ← 好 2.27 倍！确实有效")
    print("    ② 但看【全域】最坏值：")
    print("         几何 8段×32级  全域 2.96e-3（最坏点在 m=2.0）")
    print("         均匀 16段×16级 全域 2.49e-3  ← 均匀反而更好，且零成本")
    print("       → 非均匀在顶端丢了半格，把内点的优势吃掉了")
    print("    ③ 同为 9bit：")
    print("         几何 8段×32级  内点 1.68e-3（需 log2）")
    print("         均匀 16段×32级 内点 1.61e-3（零成本）← 打平甚至更好")
    print()
    print("    → 结论：把非均匀所需的【额外索引位数与定位逻辑】计入后，")
    print("            它相对【均匀加段数】不再占优。")

    # ------------------------------------------------------------------
    # ⑥ 非均匀带来的新问题：端点截断
    # ------------------------------------------------------------------
    print("\n[6] 优化引入的新问题：非均匀的端点截断更严重")
    print()
    print("    固定 8bit 索引 = m∈[1,2) 只有 256 个电平，每段固定 32 个。")
    print("    而几何最优各段【在 m 上】的宽度并不相等：")
    print()
    print(f"{'段':>3} {'段宽 h_k':>10} {'折合字节数':>12} {'固定给 32 字节':>16}")
    K = 2.0 ** (1 / 8) - 1
    for k in range(8):
        mk = 2.0 ** (k / 8)
        hk = K * mk
        print(f"{k:>3} {hk:>10.6f} {hk * 256:>12.2f} {32:>16}")
    print()
    print("    → 前 4 段需要 <32 字节，后 4 段需要 >32 字节")
    print("      （总和仍是 256，但分配必须随段宽变化）")
    print("    → 固定每段 32 字节时，最宽段（k=7，需要 42.5 字节）")
    print("      只能用 32 字节表示 42.5 字节的 m 区间 → 顶端 [1.9948, 2) 丢失")
    print("    → 这就是几何方案全域最坏误差落在 m=2.0 的原因")
    print()
    print("    这是通用教训：**修一个误差项，往往引入另一个**。")
    print("    非均匀压低了内点的插值/索引误差，却放大了端点截断。")

    # ------------------------------------------------------------------
    # ⑦ 误差配平：加段数的地板
    # ------------------------------------------------------------------
    print("\n[7] 误差配平：为什么'只加段数'会撞地板")
    print()
    print("    总误差 ≈ 插值误差 + 索引量化误差，两者要配平：")
    print("      插值误差 ≈ h²/8·max|f''| = (1/segs)²/8 × 1")
    print("      索引误差 ≈ |f'|·(1/levels)/2 = 1×(1/levels)/2")
    print()
    print(f"{'方案':>17} {'总电平':>7} {'插值误差':>11} {'索引误差':>11} "
          f"{'合计':>11}")
    print("-" * 62)
    for name, sb, fb in [("8 段 × 32 级", 3, 5), ("16 段 × 16 级", 4, 4),
                         ("32 段 × 8 级", 5, 3), ("16 段 × 32 级", 4, 5),
                         ("32 段 × 32 级", 5, 5)]:
        segs = 1 << sb
        levels = 1 << (sb + fb)
        e_interp = (1.0 / segs) ** 2 / 8
        e_index = (1.0 / levels) / 2
        print(f"{name:>20} {levels:>7} {e_interp:>11.2e} {e_index:>11.2e} "
              f"{e_interp + e_index:>11.2e}")
    print()
    print("    ★ 前 3 行：总电平都是 256，索引误差【恒为 1.95e-3】")
    print("      段数 8→32（插值误差降 16 倍），合计只从 3.9e-3 降到 2.1e-3")
    print("      —— 被【索引误差地板】卡住了")
    print("    ★ 后 2 行：加电平（9bit/10bit）才能同时压低两者")
    print()
    print("    → 提升精度的三条路径，按性价比排序：")
    print("      ① 加段数（8→16 段）：表项 9→17，误差 /4，索引零成本  ★ 首选")
    print("      ② 加电平（8→9bit）：两者同时 /2，索引仍零成本")
    print("      ③ 非均匀：只能压低索引误差的一部分，还要付定位逻辑")

    # ------------------------------------------------------------------
    # ⑧ 非均匀真正值得的场景
    # ------------------------------------------------------------------
    print("\n[8] 非均匀分段真正值得的场景：饱和型函数")
    print()
    print("    回到 [1] 的曲率变化比：")
    print("      ln  [1,2)：4 倍       → 非均匀内点只赚 2.27 倍，全域还被端点吃掉")
    print("      exp [0,1)：2 倍       → 几乎均匀，非均匀无意义")
    print("      sigmoid [0,8]：上万倍 → 饱和区极平坦，一段可顶几十段")
    print()
    print("    以 sigmoid 为例，饱和区（|x|>5）的 |f''| < 1e-3，")
    print("    用平直线近似误差已极小 —— 那里可以给【极宽】的段。")
    print("    这才是非均匀真正的用武之地：")
    print("      · 曲率变化比大（饱和型：sigmoid / tanh）")
    print("      · 且段宽差异大（最宽/最窄可达几十倍）")
    print("    而 ln 的最优段宽比只有 1.83 倍 —— 没多少可挖。")
    print()
    print("    对 ln 这类曲率温和的函数，正确做法：")
    print("      · 优先【加段数】（8→16 段，误差 /4，索引零成本）")
    print("      · 保持'移位+掩码'索引")
    print("      · 不要为了 2.27 倍内点精度去引入 log2 和端点问题")

    print("\n小结: 非均匀分段的收益由【曲率变化比】决定，不是万能优化。")
    print("  · 内点误差：非均匀确实更优（ln 好 2.27 倍）")
    print("  · 但全域最坏：端点截断把优势吃掉，均匀加段数反而更好")
    print("  · 索引位数计入后：非均匀不再占优（还要 log2）")
    print("  · 真正值得的场景：sigmoid/tanh 这类【饱和型】函数")

    print("\n附: 本 step 的两个反直觉点")
    print("  ① '理论上更优' ≠ '工程上划算'：几何网格 2^(k/8) 是数学最优解，")
    print("     但要 log2 定位 + 端点损失更大，净收益为负。")
    print("  ② '修一个误差会引入另一个'：非均匀压低了内点误差，")
    print("     却因最宽段在最右，把误差搬到了端点。")


if __name__ == "__main__":
    main()
