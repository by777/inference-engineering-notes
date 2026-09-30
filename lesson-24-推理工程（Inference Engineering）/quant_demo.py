#!/usr/bin/env python3
"""量化误差演示 —— Lesson 24 配套（对应原书 5.1 / Lesson 21-22）

三件事，用纯标准库演示（不需要 numpy）：

  1. --range        动态范围：为什么浮点格式胜过整数（离群值能不能活下来）
  2. --granularity  粒度：per-tensor / per-channel / per-block 的误差差异
  3. --compound     误差复合：为什么"有状态"的组件不能随便量化

用法：
    python3 quant_demo.py              # 三个都跑
    python3 quant_demo.py --range
    python3 quant_demo.py --granularity
    python3 quant_demo.py --compound
"""

import argparse
import math
import random

# ============================================================ 量化器实现

def _e4m3_levels() -> list[float]:
    """枚举 FP8 E4M3 能表示的全部正数幅值。

    格式：1 位符号 + 4 位指数（bias=7）+ 3 位尾数，共 16×8 = 128 个格子。
    约定：exp=0 是次正规数；exp=15 & mantissa=7 保留给 NaN。
    """
    vals = []
    for e in range(16):
        for m in range(8):
            if e == 0:
                v = (m / 8) * 2 ** -6          # 次正规（含 0）
            else:
                v = (1 + m / 8) * 2 ** (e - 7)  # 正规
            if e == 15 and m == 7:
                continue                        # NaN
            vals.append(v)
    return sorted(set(vals))


_E4M3 = _e4m3_levels()
_E4M3_MAX = max(_E4M3)


def fp8_e4m3(x: float) -> float:
    """把 x 量化到最接近的 FP8 E4M3 可表示值。"""
    if x == 0:
        return 0.0
    sign = -1.0 if x < 0 else 1.0
    a = abs(x)
    if a >= _E4M3_MAX:
        return sign * _E4M3_MAX
    # 线性扫描 128 个格子找最近（够快，也够直白）
    best = _E4M3[0]
    best_d = abs(a - best)
    for v in _E4M3:
        d = abs(a - v)
        if d < best_d:
            best, best_d = v, d
    return sign * best


def int8_symmetric(x: float, scale: float) -> float:
    """对称 int8 量化（对应 Lesson 21 的 MAX_ABS 定标法）。"""
    q = round(x / scale)
    q = max(-127, min(127, int(q)))
    return q * scale


# ============================================================ 1. 动态范围

def demo_range():
    print("=" * 76)
    print("[1] 动态范围：为什么浮点胜过整数（原书 5.1.1）")
    print("=" * 76)
    print("同一组「推理里常见的值」，分别用 int8 和 fp8_e4m3 表示，看谁活得下来。\n")

    # 典型激活分布：大部分值很小，但有几个离群值（离群值携带真实信号）
    values = [0.01, 0.05, 0.2, 0.5, 1.0, 3.0, 8.0, 20.0, 100.0]
    scale_i8 = max(values) / 127.0

    print(f"{'原值':>8} {'int8 还原':>12} {'相对误差':>10}   {'fp8 还原':>10} {'相对误差':>10}")
    print("-" * 76)
    for v in values:
        qi = int8_symmetric(v, scale_i8)
        qf = fp8_e4m3(v)
        ei = abs(qi - v) / v * 100
        ef = abs(qf - v) / v * 100
        print(f"{v:>8.3f} {qi:>12.4f} {ei:>9.1f}%   {qf:>10.4f} {ef:>9.1f}%")
    print("-" * 76)

    sub = [v for v in _E4M3 if 0 < v < 1]
    print(f"\nFP8 E4M3 能表示 {len(_E4M3)} 个正幅值（+ 0），最大值 {_E4M3_MAX:.0f}，"
          f"最小正规/次正规 {min(sub):.6f}")
    print(f"  · 最小/最大 = {min(sub)/_E4M3_MAX:.2e}  → 这就是【动态范围】很宽")
    print(f"  · int8 是均匀格子，格子宽度恒为 {scale_i8:.4f}")
    print(f"    → 小值落进同一个格子（粒度粗暴），大值附近的格子又太粗")
    print()
    print("关键认知：")
    print("  · int8 只有 256 个均匀格子，没有指数位 → 【小值被压成 0，离群值靠削顶】")
    print("  · fp8 把位分成 符号/指数/尾数 → 指数位买来动态范围 → 大小值能共存")
    print("  · 离群值在神经网络里携带真实信号，抹平它们 = 丢信号（原书的论点）")


# ============================================================ 2. 粒度

def _make_tensor(channels=64, per_channel=64, seed=7):
    """造一个带离群值的张量：模拟某几个通道里有大离群值。

    形状按"通道"组织：每个通道是一个特征向量（对应 per-channel 量化的一行）。
    """
    rnd = random.Random(seed)
    data = []
    for c in range(channels):
        row = [rnd.uniform(-0.5, 0.5) for _ in range(per_channel)]
        # 第 5 和第 40 通道里塞入离群值（真实模型中 attention/QKV 常见）
        if c in (5, 40):
            row[3] *= 60.0
            row[17] *= 45.0
        data.append(row)
    return data


def _quant_per_tensor(data):
    flat = [v for row in data for v in row]
    scale = max(abs(v) for v in flat) / 127.0
    return [[int8_symmetric(v, scale) for v in row] for row in data], scale


def _quant_per_channel(data):
    out = []
    for row in data:
        scale = max(abs(v) for v in row) / 127.0
        out.append([int8_symmetric(v, scale) for v in row])
    return out


def _quant_per_block(data, block=32):
    out = []
    for row in data:
        new_row = []
        for i in range(0, len(row), block):
            chunk = row[i:i + block]
            scale = max(abs(v) for v in chunk) / 127.0
            new_row.extend(int8_symmetric(v, scale) for v in chunk)
        out.append(new_row)
    return out


def _mse(a, b):
    n = sum(len(r) for r in a)
    s = sum((x - y) ** 2 for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    return s / n


def demo_granularity():
    print("=" * 76)
    print("[2] 粒度：多少值共享一个 scale factor（原书 5.1.1）")
    print("=" * 76)
    data = _make_tensor()
    flat = [v for row in data for v in row]

    print(f"张量形状：{len(data)} 通道 × {len(data[0])} 值 = {len(flat)} 个值")
    print(f"基础值域 [-0.5, 0.5]，但第 5、40 通道里有 2 个离群值（×60 和 ×45）")
    print(f"全局最大值 = {max(abs(v) for v in flat):.2f}\n")

    ref = _mse(data, data)
    assert ref == 0

    variants = [
        ("per-tensor (最粗，整张一个 scale)", _quant_per_tensor(data)[0]),
        ("per-channel (每通道一个 scale)   ", _quant_per_channel(data)),
        ("per-block-32 (每 32 值一个 scale) ", _quant_per_block(data, 32)),
        ("per-block-16 (NVFP4 那档粒度)    ", _quant_per_block(data, 16)),
    ]

    print(f"{'粒度':<36} {'MSE':>12} {'vs per-tensor':>14} {'被压成 0 的比例':>16}")
    print("-" * 76)
    base_mse = None
    for name, q in variants:
        m = _mse(data, q)
        if base_mse is None:
            base_mse = m
        zeros = sum(1 for row in q for v in row if v == 0.0)
        zr = zeros / len(flat) * 100
        ratio = f"{base_mse/m:.1f}× 更好" if base_mse else "基准"
        print(f"{name:<36} {m:>12.3e} {ratio:>14} {zr:>15.2f}%")
        # 顺便统计"正常通道"里的误差（离群通道之外）
        if name.startswith("per-tensor"):
            # 统计所有非离群值的平均相对误差
            errs = []
            for row, qrow in zip(data, q):
                for v, qv in zip(row, qrow):
                    if abs(v) > 1e-9:
                        errs.append(abs(qv - v) / abs(v))
            print(f"{'':<36} └ 非离群值平均相对误差 {sum(errs)/len(errs)*100:>6.1f}%"
                  f"  ← 被离群值拖累")
    print("-" * 76)
    print("判读：")
    print("  · per-tensor 最便宜，但 2 个离群值就把整张量的 scale 撑到 30/127")
    print("    → 所有正常值被挤进极少数格子 → 相对 MSE 最大")
    print("  · per-channel 救回其它通道，但【离群值所在通道内部】仍被毁（48 个值陪葬）")
    print("  · per-block 只让【包含离群值的那一个块】受损 → 误差最低")
    print("  · 粒度越细 → 离群值越局部 → 越活（这就是 MX/NVFP4 存在的理由）")
    print()
    print("代价（原书明确说了，别只记好处）：")
    print("  · 更多 scale factor 要【存】有内存开销")
    print("  · 张量级 + 块级 scale 都要【应用】，有计算开销")
    print("  · Blackwell 的做法：在 Tensor Core 里应用 scale 来抵消这部分开销")


# ============================================================ 3. 误差复合

def demo_compound():
    print("=" * 76)
    print("[3] 误差复合：为什么 「有状态」的组件不能随便量化（原书 5.1 引言）")
    print("=" * 76)

    print("\n(a) 原书的 pi 例子 —— 精度误差在幂运算下快速放大\n")
    print(f"{'pi 的精度':>12} {'pi^2':>12} {'pi^3':>12} {'pi^3 相对误差':>14}")
    print("-" * 76)
    true3 = math.pi ** 3
    for label, p in [("3.14159", 3.14159), ("3.14", 3.14), ("3", 3.0)]:
        print(f"{label:>12} {p**2:>12.6f} {p**3:>12.6f} {abs(p**3-true3)/true3*100:>13.3f}%")
    print("-" * 76)
    print("→ 同一个误差，经过单调放大后变得可观。这就是「复合」的直观形态。")

    print("\n(b) 状态回灌：模拟 KV cache / 流式模型的状态传递\n")
    print("场景：状态是一个【向量】（像每个 token 的 KV cache），每步更新后被量化，")
    print("      再作为下一步的输入。")
    print("      x_{t+1} = a * x_t + input_t   ← a 决定误差是【被吸收】还是【被放大】\n")

    def fp8_vec_quant(vec):
        """按 per-tensor 定标把向量量化到 fp8。

        真实实现（KV cache / 激活量化）一定带 scale factor：
        先由幅值范围算出 scale，把值映射到 fp8 的可用区间，再映射回来。
        不做定标直接量化标量是没有意义的（会极端失真）。
        """
        amax = max(abs(v) for v in vec)
        if amax == 0:
            return list(vec)
        scale = amax / _E4M3_MAX
        return [fp8_e4m3(v / scale) * scale for v in vec]

    steps = 200
    dim = 64

    def make_input(t):
        return [0.02 * math.sin(t * 0.1 + i * 0.05) for i in range(dim)]

    def run(a: float, quant_state: bool):
        """跑一遍带量化状态与不带量化的轨迹，返回各步的相对误差。"""
        rnd = random.Random(42)
        x = [rnd.uniform(-1, 1) for _ in range(dim)]
        ref = list(x)
        errors = {}
        for t in range(1, steps + 1):
            inp = make_input(t)
            x = [a * xi + ii for xi, ii in zip(x, inp)]
            ref = [a * ri + ii for ri, ii in zip(ref, inp)]
            if quant_state:
                x = fp8_vec_quant(x)
            if t in (20, 100, 200):
                num = sum(abs(p - q) for p, q in zip(x, ref))
                den = sum(abs(q) for q in ref)
                errors[t] = num / den * 100 if den else 0.0
        return errors

    marks = (20, 100, 200)
    print(f"{'递推系数 a':<12} {'第 20 步':>12} {'第 100 步':>12} {'第 200 步':>12}   趋势")
    print("-" * 76)
    for a, note in [(0.90, "强衰减 → 误差被吸收"), (0.98, "弱衰减 → 误差收敛"), (1.00, "不衰减 → 误差缓慢累积")]:
        e = run(a, True)
        trend = "饱和" if abs(e[200] - e[100]) < 0.25 * max(e[100], 1e-9) else "仍在增长"
        print(f"a = {a:<8.2f} {e[20]:>11.2f}% {e[100]:>11.2f}% {e[200]:>11.2f}%   {trend}（{note}）")
    print("-" * 76)

    base = run(0.98, False)
    print(f"对照：a = 0.98 但【不量化状态】 → "
          f"{base[20]:.3f}% / {base[100]:.3f}% / {base[200]:.3f}%   （全程仅舍入，无累积）")
    print("-" * 76)
    print("判读（这段比'量化会炸'更重要）：")
    print("  · 误差【是否】复合，取决于递推的放大/衰减特性，不是量化本身")
    print("  · a < 1（衰减）：每步的量化误差被按 (1-a) 的强度吸收 → 误差【饱和】在一个平台上")
    print("      但平台高度 = 单步误差 / (1-a)，衰减越弱平台越高（0.90 → 0.98 可见明显抬升）")
    print("  · a = 1（不衰减）：没有吸收机制 → 误差【持续累积】")
    print()
    print("⚠️ 所以真实系统里 attention/KV cache 危险，是因为：")
    print("   · 每个新 token 要对【全部历史】做加权求和 → 误差不只来自上一步，而是持续汇入")
    print("   · 这个求和没有强衰减（不像这里的 0.98 会把历史冲淡）")
    print("   · 而 weights 是静态的，每次用的是同一份 → 误差不累积，所以最安全")
    print()
    print("  ⇒ 一句话：**静态的可以粗量化，有状态的要小心，被长期聚合的最危险**")

    print("\n关键认知（对应你的实际工作）：")
    print("  · 敏感度排序的因果解释：")
    print("      weights   静态、每次用同一份   → 误差【不】累积  → 最不敏感")
    print("      activations 随输入变化         → 误差局部化      → 次之")
    print("      KV cache  被后续每个 token 复用 → 误差持续汇入    → 中等敏感")
    print("      attention 既敏感又层层依赖      → 最危险")
    print("  · 【你的对应场景】流式降噪 的 next_state 回灌就是「有状态」：")
    print("      关键在于它的递推是【衰减还是放大】——")
    print("      衰减型 → 误差会饱和，量化后可接受")
    print("      放大/中性 → 误差持续累积，必须专门验证长序列（不能只看单帧指标）")
    print("  · 【KWS 也是】流式场景的状态若量化，误唤醒率漂移可能只在长音频上暴露")


def main():
    p = argparse.ArgumentParser(description="量化误差演示（Lesson 24 配套）")
    p.add_argument("--range", action="store_true", help="动态范围：int8 vs fp8")
    p.add_argument("--granularity", action="store_true", help="粒度：per-tensor/channel/block")
    p.add_argument("--compound", action="store_true", help="误差复合与状态回灌")
    args = p.parse_args()

    any_flag = args.range or args.granularity or args.compound
    if args.range or not any_flag:
        demo_range()
        print()
    if args.granularity or not any_flag:
        demo_granularity()
        print()
    if args.compound or not any_flag:
        demo_compound()


if __name__ == "__main__":
    main()
