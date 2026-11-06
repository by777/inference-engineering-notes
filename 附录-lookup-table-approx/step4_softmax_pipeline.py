#!/usr/bin/env python3
"""
step4: 完整 softmax 定点流水 —— 查表法在真实算子里怎么串起来

一条完整的定点 softmax 流水（对每个 logit x_i）：
  ① x_max = max(x)                       # 数值稳定
  ② t_i   = (x_i - x_max) · log2e        # 换底：exp → 2 的幂，结果 ≤ 0
  ③ 拆 t_i = n_i + f_i                   # n_i 整数(≤0)、f_i∈[0,1)
  ④ e_i   = 2^f_i 查表插值，再按 n_i 移位还原 2^n
  ⑤ 分母 S = Σ e_j，p_i = e_i / S

本 step 用 Python 完整实现上面每条（定点全部用整数运算模拟，
定点小数用 Q12 表示），并和浮点真值 softmax 对拍误差。
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lut_common import LOG2E, Q, gen_exp2_table, exp2_lookup_lerp


def fixed_softmax(logits):
    """定点 softmax：内部 exp 用 9 项表 + 5bit 插值，全程整数。

    返回 (p_approx, debug)：p_approx 是查表近似流程算出的概率。
    注意：p_approx 已是 float —— 归一化那步 e/S 用了 Python 真除法，
    同标尺相除 4096 自动约掉，所以结果直接是 0~1 的概率，不再是定点整数。
    debug 是每步的中间量，便于讲解。
    """
    table = gen_exp2_table()          # Q12
    xmax = max(logits)
    n_list, f_list, fb_list, e_list, t_list = [], [], [], [], []
    for x in logits:
        t = (x - xmax) * LOG2E        # ≤ 0
        # 拆整数 n（向 -inf 取整，保证 f∈[0,1)）
        n = math.floor(t)
        f = t - n
        f_byte = min(255, int(round(f * 256)))   # f→8bit
        # 查表 + 插值得到 2^f 的 Q12 值
        e_q12 = exp2_lookup_lerp(f_byte, table)
        # 还原 2^n：n 是整数，Q12 域里左移 n 位；n≤0 → 右移
        # 注意 n 可能很负（动态范围大），右移过多会变 0 —— 见 step5/roadmap 的 block-float
        if n >= 0:
            e = e_q12 << n
        else:
            e = e_q12 >> (-n)
        n_list.append(n); f_list.append(f)
        fb_list.append(f_byte); e_list.append(e); t_list.append(t)
    S = sum(e_list)
    # e/S：分子分母同为 Q12，4096 自动约掉；Python 的 / 是真除法 → 结果已是 float 概率
    p_approx = []
    for e in e_list:
        p_approx.append(e / S)
    debug = dict(t=t_list, n=n_list, f=f_list, fb=fb_list, e=e_list, S=S)
    return p_approx, debug


def float_softmax(logits):
    xmax = max(logits)

    exps = []
    for x in logits:
        exps.append(math.exp(x - xmax))

    s = sum(exps)

    p = []
    for e in exps:
        p.append(e / s)
    return p


def main():
    print("=" * 72)
    print("step4  完整 softmax 定点流水（内部 exp = 9 项表 + 5bit 插值）")
    print("=" * 72)

    logits = [3.2, 1.5, 0.0, -2.0, -5.0]    # 真实一点的 logits（跨 8.2）

    print(f"\n[1] logits = {logits}")
    p_approx, dbg = fixed_softmax(logits)
    p_true = float_softmax(logits)

    print(f"\n[2] 定点流水中途量（逐元素）")
    print(f"{'x':>6} {'t=(x-m)·log2e':>14} {'n':>4} {'f':>8} {'f_byte':>7} {'2^f表值Q12':>10} {'e(Q12→shift)':>12}")
    for i, x in enumerate(logits):
        print(f"{x:>6.2f} {dbg['t'][i]:>14.4f} {dbg['n'][i]:>4d} {dbg['f'][i]:>8.4f} "
              f"{dbg['fb'][i]:>7d} {dbg['e'][i]/(1<<Q) if dbg['n'][i]>=0 else '>>'+str(-dbg['n'][i]):>10} "
              f"{dbg['e'][i]:>12d}")

    print(f"\n[3] 分母 S = {dbg['S']} (Q12 域整数和)")

    print(f"\n[4] 概率对拍")
    print(f"{'x':>6} {'p_approx':>12} {'p_true':>12} {'Δp':>12}")
    maxdp = 0.0
    for i, x in enumerate(logits):
        dp = abs(p_approx[i] - p_true[i])
        maxdp = max(maxdp, dp)
        print(f"{x:>6.2f} {p_approx[i]:>12.6f} {p_true[i]:>12.6f} {dp:>12.2e}")
    print(f"\n    max |Δp| = {maxdp:.2e}")
    print(f"    argmax: approx={logits[p_approx.index(max(p_approx))]}, true={logits[p_true.index(max(p_true))]}")
    print(f"    softmax 误差被分母归一化摊薄：exp 相对误差 ~1e-3 → 概率误差 ~1e-4")

    # ---------- 动态范围测试 ----------
    print("\n[5] 动态范围压力测试：logits 跨度大时定点 exp 会怎样")
    wide = [3.2, -20.0]      # 跨度 23.2
    pfw, dbgw = fixed_softmax(wide)
    ptw = float_softmax(wide)
    print(f"    logits = {wide}")
    print(f"    定点: 较小的项 t={dbgw['t'][1]:.2f}, n={dbgw['n'][1]}, e={dbgw['e'][1]}")
    print(f"          exp(-23.2) ≈ {math.exp(-23.2):.2e}，Q12 域 << 1 → 右移 {dbgw['n'][1]*-1} 位后直接变 0")
    print(f"    → 概率固定输出 0，真实概率 {ptw[1]:.2e} —— 对 argmax/分类无损，")
    print(f"      但对需要小概率的场合(如 KL/熵)失真 → 需 block floating point（见 roadmap）")

    print("\n小结: 查表 softmax = 减max → 乘log2e → 拆n/f → 查表插值 → 移位求和 → 归一。")
    print("9 项表、18 字节、纯整数，概率误差 ~1e-4 量级。")


if __name__ == "__main__":
    main()
