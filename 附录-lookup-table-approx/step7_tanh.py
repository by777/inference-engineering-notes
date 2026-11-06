#!/usr/bin/env python3
"""
step7: tanh 查表 —— 自己建表 vs 复用 sigmoid，实测对比

tanh(x) = 2·σ(2x) - 1   （代数关系，可复用 sigmoid 表）

核心教学点（一个反直觉的结论）：
  tanh 自己建表并不比复用 sigmoid 好多少，几乎打平。

原因：误差 ∝ 段宽² × 曲率，两个量都会变：
  · tanh 饱和快 → 阈值 4（vs sigmoid 的 8）→ 段宽减半 → 误差 /4（变好）
  · 但 tanh 曲率 max|f''|=0.77，是 sigmoid 的 0.096 的 8 倍（变差）
  · 净效果 ≈ 1/4 × 8 = 2 倍变差，把段宽优势吃掉了

本 step 展示：
  ① tanh 自己建表（X_MAX=4，奇函数砍半）
  ② 复用 sigmoid 表（tanh=2σ(2x)-1）
  ③ 实测两者误差，几乎打平
  ④ 用 ε ∝ h²·|f''| 解释"为什么打平"
  ⑤ 指令数对比（每次查表的 ALU 开销）
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def tanh_true(x):
    return math.tanh(x)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ---------------- 方案A：tanh 自己建表 ----------------
def gen_tanh_table(xmax=4.0, segs=8, q=12):
    h = xmax / segs                    # 段宽
    scale = 1 << q                     # 定点标尺 2^q

    table = []
    for k in range(segs + 1):          # 8 段 = 9 个端点
        x_k = k * h                    # 第 k 个结点的 x 值
        value = math.tanh(x_k)         # 该点的函数真值 tanh(x_k)
        table.append(round(value * scale))   # 转定点 + 四舍五入
    return table


def tanh_self_lookup(x, table, xmax=4.0):
    if x < 0:
        return -tanh_self_lookup(-x, table, xmax)   # 奇函数
    if x > xmax:
        return 1.0
    fb = min(255, int(round(x / xmax * 256)))
    idx = fb >> 5
    frac = fb & 31
    lo, hi = table[idx], table[idx + 1]
    return ((lo * (32 - frac) + hi * frac) >> 5) / 4096.0


# ---------------- 方案B：复用 sigmoid 表 ----------------
def gen_sigmoid_table(xmax=8.0, segs=8, q=12):
    h = xmax / segs                    # 段宽
    scale = 1 << q                     # 定点标尺 2^q

    table = []
    for k in range(segs + 1):          # 8 段 = 9 个端点
        x_k = k * h                    # 第 k 个结点的 x 值
        value = sigmoid(x_k)           # 该点的函数真值 σ(x_k)
        table.append(round(value * scale))   # 转定点 + 四舍五入
    return table


def sigmoid_lookup(x, table, xmax=8.0):
    if x < 0:
        return 1.0 - sigmoid_lookup(-x, table, xmax)   # 对称性
    if x > xmax:
        return 1.0
    fb = min(255, int(round(x / xmax * 256)))
    idx = fb >> 5
    frac = fb & 31
    lo, hi = table[idx], table[idx + 1]
    return ((lo * (32 - frac) + hi * frac) >> 5) / 4096.0


def tanh_reuse_lookup(x, table):
    return 2.0 * sigmoid_lookup(2.0 * x, table) - 1.0


def scan_max_err(fn, lo=-8.0, hi=8.0, n=16001):
    mx, wx = 0.0, 0.0
    for i in range(n + 1):
        x = lo + (hi - lo) * i / n
        e = abs(fn(x) - tanh_true(x))
        if e > mx:
            mx, wx = e, x
    return mx, wx


def main():
    print("=" * 72)
    print("step7  tanh 查表：自己建表 vs 复用 sigmoid")
    print("=" * 72)

    # ---------- ① 恒等性 ----------
    print("\n[1] 恒等性: tanh(x) = 2σ(2x) - 1")
    for x in [-2.0, -0.5, 0.5, 2.0]:
        ok = abs(tanh_true(x) - (2 * sigmoid(2 * x) - 1)) < 1e-15
        print(f"    x={x:>5}  {'✓' if ok else '✗'}")

    # ---------- ② 两个方案各自建表 ----------
    print("\n[2] 两个方案")
    table_self = gen_tanh_table(xmax=4.0)       # A: tanh 自己，阈值 4
    table_sig = gen_sigmoid_table(xmax=8.0)     # B: sigmoid 表，阈值 8
    print(f"    A 自己建表: tanh, X_MAX=4, {len(table_self)} 项")
    print(f"    B 复用sigmoid: σ, X_MAX=8, {len(table_sig)} 项 + 2σ(2x)-1")

    # ---------- ③ 实测误差 ----------
    print("\n[3] 实测最大误差（扫 x∈[-8,8]）")
    e_self, wx_self = scan_max_err(lambda x: tanh_self_lookup(x, table_self))
    e_reuse, wx_reuse = scan_max_err(lambda x: tanh_reuse_lookup(x, table_sig))
    print(f"    A 自己建表   max_err = {e_self:.5f}  @ x={wx_self:.3f}")
    print(f"    B 复用sigmoid max_err = {e_reuse:.5f}  @ x={wx_reuse:.3f}")
    print(f"    → 几乎打平（A 只比 B 好 {e_reuse/e_self:.2f} 倍）")

    # ---------- ④ 用公式解释 ----------
    print("\n[4] 为什么打平：ε ∝ 段宽² × 曲率，两个量对冲")
    f2pp_sig = 0.0962   # sigmoid max|f''|
    f2pp_tanh = 0.7698  # tanh   max|f''|（x≈0.658）
    h_sig, h_tanh = 8.0 / 8, 4.0 / 8    # 段宽 1.0 vs 0.5
    print(f"    sigmoid: h={h_sig}, max|f''|={f2pp_sig}")
    print(f"    tanh:    h={h_tanh}, max|f''|={f2pp_tanh}")
    print(f"    段宽比 tanh/sigmoid = {h_tanh/h_sig} → h² 贡献 {(h_tanh/h_sig)**2} 倍（变好）")
    print(f"    曲率比 tanh/sigmoid = {f2pp_tanh/f2pp_sig:.1f} 倍（变差）")
    print(f"    净效果 ≈ {(h_tanh/h_sig)**2} × {f2pp_tanh/f2pp_sig:.1f} = {(h_tanh/h_sig)**2*f2pp_tanh/f2pp_sig:.2f} 倍 → 打平")

    # ---------- ⑤ 指令数对比 ----------
    print("\n[5] 指令数对比（每次查表的 ALU 开销）")
    print("    A 自己建表:")
    print("       cmp(x<0) → cmp(x>4) → x<<6(定点化) → 拆idx/frac → 访存 → 插值(2乘加+1移)")
    print("    B 复用sigmoid(2σ(2x)-1):")
    print("       x<<1(算2x) → cmp → cmp → 定点化 → 拆位 → 访存 → 插值 → ×2-1")
    print("    B 比 A 多 3 条 ALU：x<<1、×2、-1")
    print("    → 单看 cycle：A 自己建表更优（少 3 条指令/次）")

    print("\n小结: 三维权衡 ——")
    print("    精度: 打平（0.0279 vs 0.0281）")
    print("    cycle: 自己建表优（少 3 条指令）")
    print("    存储: 复用省 18 字节")
    print("    → 只要 tanh：自己建表更优；系统已有 sigmoid 表：复用才划算")
    print("    教训：不能只看精度或只看存储，精度/cycle/存储 三维都要摆出来。")


if __name__ == "__main__":
    main()
