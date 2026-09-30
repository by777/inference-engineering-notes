#!/usr/bin/env python3
"""
step13: softmax 分母 1/S 的定点除法 —— 补上最后一块拼图

════════════════ 为什么需要这一步 ════════════════

step4 的完整 softmax 流水到最后一行为止是这样的：

    S = Σ e_j
    p_i = e_i / S          ← ⚠️ 这里是 Python 浮点除法

前面所有步骤（range reduction、建表、插值、移位还原）都用整数模拟，
唯独这最后一步用了浮点除法 —— 在无 FPU 的硬件上这行不可用。
本 step 把它也定点化，让 softmax 真正【全程无除法】。

════════════════ 方案 ════════════════

要算 p_i = e_i / S，只需求出 1/S，然后乘上去：

    p_i = e_i · (1/S)

而 1/S 用 step8 的老办法（range reduction + 查倒数表 + 牛顿迭代）：

    ① S = m·2^e                     （S 是整数，用 bit_length 拆）
    ② 查表 + 插值 + 牛顿 → 1/m 的 Q30 定点值 y
    ③ 1/S = y · 2^(-30-e)
    ④ p_i = e_i · y · 2^(-30-e)

合并移位：p_i 要 Q12 结果，则
    p_i_q12 = (e_i · y_q30) >> (18 + e)

本 step 展示：
  ① step4 的缺口（唯一一处浮点）
  ② 为什么 S 必须 range reduction（动态范围宽）
  ③ 零点乘法的索引（和 step11 的 ln 同源）
  ④ 牛顿轮数的选择（2 轮就到底）
  ⑤ 全整数 softmax 对拍
  ⑥ ⚠️ 截断偏差：Σp 会系统性 < 1，以及怎么修
  ⑦ 开销对比：真除法 vs 倒数+乘法
"""
import math
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent))
from step4_softmax_pipeline import fixed_softmax, float_softmax

Q = 12            # e_i / p_i 的定点位数（与 step4 一致）
Q_RCP = 30        # 倒数内部精度：Q30（为什么不用 Q12，见 [4]）


# ---------------------------------------------------------------------------
# ② 倒数表 + range reduction + 牛顿（全程整数）
# ---------------------------------------------------------------------------
def gen_rcp_table(segs=8, q=Q):
    """对 m∈[1,2) 建 1/m 表（Q12）。

    和 step8 完全同一张表：结点 m_k = 1+k/8，表项 = round((1/m_k)·2^q)。
    存 Q12 就够 —— 它只负责给牛顿迭代一个"不太远"的初值。
    """
    width = 1.0                        # 区间 [1,2) 宽 1
    scale = 1 << q

    table = []
    for k in range(segs + 1):          # 8 段 = 9 个端点
        m_k = 1.0 + k / segs * width
        value = 1.0 / m_k
        table.append(round(value * scale))
    return table


def split_int(S):
    """把正整数 S 拆成 m·2^e，m∈[1,2)，m 用 Q30 整数返回。

    注意这里【不用 math.frexp】—— 那会先转成 float，S 大时丢精度，
    而且板上也没有 frexp。改用 bit_length 定位最高位，等价且精确：
        2^e ≤ S < 2^(e+1)   ⟺   e = S.bit_length() - 1
    """
    e = S.bit_length() - 1
    if e <= Q_RCP:
        m_q30 = S << (Q_RCP - e)        # m_q30 ∈ [2^30, 2^31)
    else:
        m_q30 = S >> (e - Q_RCP)        # 极端大的 S（实际到不了）
    return m_q30, e


def rcp_q30(S, table, rounds=2):
    """求 1/S 的 Q30 定点值，全整数。返回 (y_q30, e)。

    不直接返回 1/S 的数值 —— 因为 1/S = y·2^(-30-e)，
    带一个指数 e 更利于后面合并到一个移位里（避免中途丢精度）。
    """
    m_q30, e = split_int(S)

    # ---- 索引：m∈[1,2) → 8bit，和 step11 的 ln 一样是【零点乘法】 ----
    # (m-1)·256，在 Q30 域里就是 (m_q30 - 2^30) 右移 22 位（2^30/256 = 2^22）
    # 加 2^21 是四舍五入（半 LSB）
    offset = m_q30 - (1 << Q_RCP)
    f_byte = (offset + (1 << (Q_RCP - 8 - 1))) >> (Q_RCP - 8)
    if f_byte > 255:
        f_byte = 255
    idx = f_byte >> 5                  # 高 3bit → 段号
    frac = f_byte & 31                 # 低 5bit → 段内位置

    # ---- 插值得 1/m 的 Q12 初值，升到 Q30 ----
    lo, hi = table[idx], table[idx + 1]
    y0_q12 = (lo * (32 - frac) + hi * frac) >> 5
    y = y0_q12 << (Q_RCP - Q)          # Q12 → Q30

    # ---- 牛顿迭代：y ← y·(2 - m·y)，二次收敛 ----
    one_q30 = 1 << Q_RCP
    two_q30 = 1 << (Q_RCP + 1)
    for _ in range(rounds):
        my_q30 = (m_q30 * y) >> Q_RCP   # m·y 的 Q30
        t_q30 = two_q30 - my_q30        # (2 - m·y) 的 Q30
        y = (y * t_q30) >> Q_RCP

    return y, e


# ---------------------------------------------------------------------------
# 完整整数 softmax
# ---------------------------------------------------------------------------
def softmax_fixed_div(logits, rounds=2, use_rounding=False):
    """全整数 softmax：exp 复用 step4 的流水，只把最后的除法换成倒数乘法。

    返回 (p_q12, debug)。
    """
    # 复用 step4 的 exp 流水，拿到同一批 e_i（这样对比才是公平的：
    # 只有"除法"这一处不同）
    _p_from_step4, dbg_raw = fixed_softmax(logits)
    # step4 的 debug 字典是【异构】的（既有 list 又有 int），
    # Pylance 从 dict(...) 推不出精确类型，这里显式声明，避免误报
    dbg = cast("dict[str, Any]", dbg_raw)
    e_list = dbg["e"]
    S = dbg["S"]

    table = gen_rcp_table()
    y_q30, eS = rcp_q30(S, table, rounds=rounds)

    # p_i_q12 = (e_i · y_q30) >> (18 + eS)
    #   推导：p_i = e_i/S = e_i · (y_q30/2^30) · 2^-eS
    #         要 Q12 结果就再 ×2^12
    #         → 4096·e_i·y_q30/2^(30+eS) = (e_i·y_q30) >> (18+eS)
    shift = (Q_RCP - Q) + eS           # = 18 + eS
    half = 1 << (shift - 1)
    p_q12 = []
    for ei in e_list:
        prod = ei * y_q30
        if use_rounding:
            prod += half               # 四舍五入而不是截断
        p_q12.append(prod >> shift)

    debug = dict(e=e_list, S=S, y_q30=y_q30, eS=eS, shift=shift)
    return p_q12, debug


# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("step13  softmax 分母 1/S 的定点除法")
    print("=" * 74)

    # ------------------------------------------------------------------
    # ① step4 的缺口
    # ------------------------------------------------------------------
    print("\n[1] 先看 step4 的缺口：最后一步是浮点除法")
    print()
    print("    step4 的流水：")
    print("      ① x_max = max(x)")
    print("      ② t_i = (x_i - x_max)·log2e          ← 浮点乘（可用定点替换）")
    print("      ③ 拆 t_i = n_i + f_i                 ← 整数")
    print("      ④ e_i = 查表插值(2^f_i) 再 <<n_i      ← 纯整数 ✓")
    print("      ⑤ S = Σ e_j                          ← 纯整数 ✓")
    print("      ⑥ p_i = e_i / S                      ← ⚠️ 浮点除法")
    print()
    print("    前五步都是整数，只有第 ⑥ 步漏了。本 step 补它。")

    # ------------------------------------------------------------------
    # ② 为什么 S 需要 range reduction
    # ------------------------------------------------------------------
    print("\n[2] 为什么不能直接对 S 查表？—— S 的动态范围太宽")
    print()
    print(f"{'logits 个数 N':>14} {'S 大致范围':>20} {'e 大致范围':>16}")
    for N in [4, 16, 64, 256, 1024]:
        lo = 4096                     # 最大项贡献 4096（Q12 的 1.0）
        hi = N * 4096
        lo_e = lo.bit_length() - 1
        hi_e = hi.bit_length() - 1
        s_range = "[" + str(lo) + ", " + str(hi) + "]"
        e_range = "[" + str(lo_e) + ", " + str(hi_e) + "]"
        print(f"{N:>14} {s_range:>20} {e_range:>16}")
    print()
    print("    倒数表只覆盖 m∈[1,2)（9 项），而 S 可以跨好几个数量级。")
    print("    → 必须先 range reduction：S = m·2^e，把 m 压回 [1,2) 再查表。")
    print("    这和 step8 处理任意 x 的办法完全一样，只是 S 现在是整数。")

    # ------------------------------------------------------------------
    # ③ 索引零乘法
    # ------------------------------------------------------------------
    print("\n[3] 附带的好处：索引是零点乘法（和 step11 的 ln 同源）")
    print()
    print("    因为表域是 [1,2)（宽度 1），归一化不需要除法：")
    print("      f_byte = (m-1)·256")
    print("    在 Q30 域里 '乘 256' = 右移 22 位（2^30/256 = 2^22）：")
    print("      f_byte = (m_q30 - 2^30 + 2^21) >> 22")
    print()
    print("    对比 rsqrt 的表域 [1,4)（宽度 3）：索引要除 3，")
    print("    不是 2 的幂，必须用魔法数乘法 ((m_q15-32768)*21845)>>23。")
    print("    → 表域选 [1,2) 就白拿了这个零成本索引。")

    # ------------------------------------------------------------------
    # ④ 牛顿轮数
    # ------------------------------------------------------------------
    print("\n[4] 倒数精度：为什么内部用 Q30，以及牛顿要几轮")
    print()
    print("    ① 为什么内部 Q30 而不是 Q12？")
    print("       最终要算 (e_i·y) >> (18+e)。若 y 只有 Q12 精度（~2.4e-4），")
    print("       它的量化噪声会直接成为 p_i 的相对误差。")
    print("       要 p_i 达到 1e-6 量级，y 得先有 1e-9 量级 → Q30（2^-30≈9.3e-10）。")
    print()
    table = gen_rcp_table()
    print(f"    {'牛顿轮数':>8} {'max 相对误差':>16} {'说明':>26}")
    print("    " + "-" * 54)
    for r in [0, 1, 2, 3]:
        mx = 0.0
        for S in range(4096, 40000, 7):
            y, eS = rcp_q30(S, table, rounds=r)
            est = y / 2 ** Q_RCP * 2.0 ** (-eS)
            mx = max(mx, abs(est - 1.0 / S) / (1.0 / S))
        note = {0: "只有查表", 1: "快了但没到底",
                2: "已到 Q30 地板", 3: "再多也没用"}[r]
        print(f"    {r:>8} {mx:>16.3e} {note:>26}")
    print()
    print("    → 平方收敛：5.19e-3 → 2.70e-5 → 1.76e-9 → 9.31e-10(地板)")
    print("      两轮足够；第三轮已经撞到 Q30 量化地板。")

    # ------------------------------------------------------------------
    # ⑤ 精度
    # ------------------------------------------------------------------
    print("\n[5] 倒数本身有多准（随便挑几个 S）")
    print()
    print(f"{'S':>9} {'e':>3} {'y_q30':>11} {'est 1/S':>16} "
          f"{'true 1/S':>16} {'rel err':>10}")
    for S in [4096, 5000, 8191, 12345, 65536, 100000]:
        y, eS = rcp_q30(S, table, rounds=2)
        est = y / 2 ** Q_RCP * 2.0 ** (-eS)
        truth = 1.0 / S
        rel = abs(est - truth) / truth
        print(f"{S:>9} {eS:>3} {y:>11} {est:>16.10e} {truth:>16.10e} "
              f"{rel:>10.2e}")
    print()
    print("    → 误差都 ≤ 1e-9（Q30 地板）。S 是 2 的幂时精确为 0。")

    # ------------------------------------------------------------------
    # ⑥ 完整 softmax 对拍
    # ------------------------------------------------------------------
    print("\n[6] 全整数 softmax 对拍")
    print()
    tests = [
        ("等距 4 项", [1.0, 2.0, 3.0, 4.0]),
        ("单峰 5 项", [5.0, 1.0, 0.5, 0.1, -2.0]),
        ("宽跨度 6 项", [10.0, 5.0, 0.0, -5.0, -10.0, -20.0]),
        ("接近均匀 8 项", [1.0, 1.01, 1.02, 0.99, 1.0, 1.0, 1.0, 1.0]),
    ]
    print(f"{'算例':>14} {'S':>7} {'eS':>4} {'Σp':>10} "
          f"{'vs 浮点':>11} {'vs step4':>11}")
    print("-" * 74)
    for name, logits in tests:
        p_q12, dbg = softmax_fixed_div(logits, rounds=2)
        p_int = [v / 4096.0 for v in p_q12]
        p_ref = float_softmax(logits)
        # step4 用同一批 e_i，只是除法用浮点
        p_step4 = [v / dbg["S"] for v in dbg["e"]]

        d_ref = max(abs(a - b) for a, b in zip(p_int, p_ref))
        d_s4 = max(abs(a - b) for a, b in zip(p_int, p_step4))
        print(f"{name:>14} {dbg['S']:>7} {dbg['eS']:>4} {sum(p_int):>10.6f} "
              f"{d_ref:>11.3e} {d_s4:>11.3e}")
    print()
    print("    'vs step4' 一列隔离出【纯除法】引入的误差 —— 两者用的是同一批 e_i。")
    print("    可以看到：定点除法的贡献和 exp 查表误差同量级（~1e-4），")
    print("    也就是说它【不是新瓶颈】，但也不是白拿。")

    # ------------------------------------------------------------------
    # ⑦ 截断偏差
    # ------------------------------------------------------------------
    print("\n[7] ⚠️ 截断偏差：Σp 会系统性小于 1")
    print()
    print("    每项 p_i = (e_i·y) >> shift 都是【向下截断】，")
    print("    所以每个 p_i 都偏小一点，累加起来 Σp < 1。")
    print()
    print(f"{'算例':>14} {'截断 Σp':>11} {'舍入 Σp':>11} "
          f"{'截断误差':>11} {'舍入误差':>11}")
    print("-" * 66)
    for name, logits in tests:
        p_ref = float_softmax(logits)
        p_tr, _ = softmax_fixed_div(logits, rounds=2, use_rounding=False)
        p_rd, _ = softmax_fixed_div(logits, rounds=2, use_rounding=True)
        s_tr = sum(v / 4096.0 for v in p_tr)
        s_rd = sum(v / 4096.0 for v in p_rd)
        d_tr = max(abs(v / 4096.0 - r) for v, r in zip(p_tr, p_ref))
        d_rd = max(abs(v / 4096.0 - r) for v, r in zip(p_rd, p_ref))
        print(f"{name:>14} {s_tr:>12.6f} {s_rd:>14.6f} "
              f"{d_tr:>12.3e} {d_rd:>14.3e}")
    print()
    print("    → 四舍五入让 Σp 精确回到 1（偏差被消掉），")
    print("      但【单点误差】只是被重新分配：有可能变小、也可能变大。")
    print("      原因：半 LSB = 1/4096 = 2.44e-4，与误差本身同量级，")
    print("      所以每个 p_i 会被推动 ±1 个 LSB —— 方向不定。")
    print()
    print("      看上面数据：等距/宽跨度变好，单峰变差，接近均匀持平。")
    print()
    print("    → 这是【偏差 vs 方差】的典型取舍：")
    print("      · 截断：偏差大（Σp 系统性偏小 ~1e-3），单点误差小")
    print("      · 四舍五入：偏差≈0（Σp 精确为 1），单点误差重新分配")
    print()
    print("    怎么选取决于下游：")
    print("      · 只关心 argmax → 截断就行（单调性不受偏差影响）")
    print("      · 要当概率用（加权求和、交叉熵）→ 用四舍五入，Σp 必须准")
    print("      · 更讲究的做法：截断后再把余量补回最大项（保 Σp=1 且单调）")

    # ------------------------------------------------------------------
    # ⑧ 开销
    # ------------------------------------------------------------------
    print("\n[8] 开销对比：真除法 vs 倒数+乘法")
    print()
    print(f"{'做法':>18} {'操作':>26} {'每项':>12}")
    print("-" * 60)
    print(f"{'① 逐项真除法 e_i/S':>18} {'N 次整数除法':>26} {'20-40 周期':>12}")
    print(f"{'② 倒数 + 乘法':>18} {'1 次倒数，再 N 次乘法':>26} {'约 1 次乘法':>12}")
    print()
    print("    设 N=1024 项：")
    print(f"      ① N 次除法  ≈ 1024 × 30 = {1024*30:>7} 周期")
    print(f"      ② 倒数一次  ≈ 30 (查表) + 2×6 (牛顿) = 42 周期")
    print(f"         再 N 次乘法 ≈ 1024 × 1 = 1024 周期")
    print(f"         合计 ≈ {42 + 1024:>7} 周期  → 约快 {1024*30/(42+1024):.1f} 倍")
    print()
    print("    ★ 关键：把【N 次昂贵的除法】换成【1 次倒数 + N 次便宜乘法】")
    print("      这是'计算强度'的经典操作 —— 分母被 N 项共享，所以只算一次。")

    print("\n小结: 定点除法的做法 = range reduction + 查倒数表 + 牛顿 + 乘法。")
    print("  · 和 step8 同一套办法，只是输入从任意 float 变成整数 S")
    print("  · 索引是零点乘法（表域 [1,2) 白拿的好处）")
    print("  · 内部用 Q30，牛顿 2 轮到底")
    print("  · 意外收获：截断 vs 四舍五入 = 偏差 vs 方差的取舍")
    print("  → 至此 softmax 真正【全程无除法】。")


if __name__ == "__main__":
    main()
