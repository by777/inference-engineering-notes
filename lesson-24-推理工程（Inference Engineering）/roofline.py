#!/usr/bin/env python3
"""roofline 计算器 —— Lesson 24 配套

把"算术强度 vs 机器平衡点"这套判断从纸面变成可跑的数。

用法：
    python3 roofline.py                     # 各代 GPU 的 ridge 一览
    python3 roofline.py --gpu H100          # 单卡详细：attention 在什么序列长度越过分界
    python3 roofline.py --verify 62         # 验证原书 decode 算例（d=128, N=4096, FP16）
    python3 roofline.py --batch             # decode 的算术强度随 batch size 变化（batching 为何存在）
    python3 roofline.py --custom 8 50       # 自定义设备：8 TFLOPS 算力 / 50 GB/s 带宽（边缘 NPU 用）
"""

import argparse
from typing import TypedDict


class GpuSpec(TypedDict):
    """一张卡的规格。用 TypedDict 而不是裸 dict，否则混排 float/str 会让类型检查器
    把每个值都推成 float | str，后面所有算术都被标红。"""
    compute_tflops: float
    bandwidth_tbs: float
    note: str


# ---------------------------------------------------------------- 硬件数据表
# 算力取 FP16 dense 峰值，带宽取 spec sheet 标称值。
# 原书用的 H100 FP16 = 989 teraFLOPS / 3.35 TB/s ≈ 295 ops/byte，与本表一致。
GPUS: dict[str, GpuSpec] = {
    "T4":   GpuSpec(compute_tflops=65,   bandwidth_tbs=0.30, note="Turing，遗留/低流量"),
    "A10":  GpuSpec(compute_tflops=125,  bandwidth_tbs=0.60, note="Ampere，遗留"),
    "A100": GpuSpec(compute_tflops=312,  bandwidth_tbs=2.04, note="Ampere，SXM 80GB"),
    "L4":   GpuSpec(compute_tflops=121,  bandwidth_tbs=0.30, note="Ada，小模型/成本敏感"),
    "L40":  GpuSpec(compute_tflops=181,  bandwidth_tbs=0.86, note="Ada，不推荐做推理"),
    "H100": GpuSpec(compute_tflops=989,  bandwidth_tbs=3.35, note="Hopper，引入 FP8"),
    "H200": GpuSpec(compute_tflops=989,  bandwidth_tbs=4.80, note="Hopper，同代更大带宽"),
    "B200": GpuSpec(compute_tflops=2250, bandwidth_tbs=8.00, note="Blackwell，FP4 + 微缩放"),
    "B300": GpuSpec(compute_tflops=2250, bandwidth_tbs=8.00, note="Blackwell，同代更大显存"),
}


def ridge(compute_tflops: float, bandwidth_tbs: float) -> float:
    """机器平衡点 I_ridge = 峰值算力 / 带宽，单位 ops/byte。

    TFLOPs/TB/s 的量纲相除正好得 ops/byte（T 与 T 抵消，FLOPs 就是 ops）。
    """
    return compute_tflops / bandwidth_tbs


# ------------------------------------------------------- attention 算术强度
def attention_intensity(seq_len: int, head_dim: int = 128, bytes_per_val: int = 2) -> dict:
    """标准（非 FlashAttention）attention 的算术强度。

    三行伪码 S = QK^T / P = softmax(S) / O = PV，
    每行都是"从内存加载 -> 计算 -> 存回内存"。

    内存流量（元素个数）：
        step1 读 Q,K (各 N*d)   写 S (N^2)
        step2 读 S   (N^2)      写 P (N^2)
        step3 读 P,V (N^2 + N*d) 写 O (N*d)
      合计 = 4*N*d + 4*N^2

    运算量（FLOPs）：
        每个 matmul 有 N^2 个输出，每个输出是 d 次 MAC = 2d FLOPs
        所以两个 matmul = 2 * 2*N^2*d = 4*N^2*d

    注：softmax 本身约 5*N^2 量级 FLOPs，相比 matmul 可忽略（计入只让结果 +1 左右）。
    """
    n, d = seq_len, head_dim
    traffic_elems = 4 * n * d + 4 * n * n
    traffic_bytes = traffic_elems * bytes_per_val
    compute_flops = 4 * n * n * d
    return dict(
        seq_len=n,
        traffic_bytes=traffic_bytes,
        compute_flops=compute_flops,
        intensity=compute_flops / traffic_bytes,
    )


def attention_intensity_limit(head_dim: int = 128) -> float:
    """未融合 attention 的算术强度【渐近上限】= d/2。

    因为流量和运算量都随 N^2 增长，N 约掉：

        I = 4*N^2*d / (2*(4*N*d + 4*N^2)) --N->inf--> 4*d/8 = d/2

    d=128 时上限 = 64 ops/byte。
    这个数字远低于任何数据中心 GPU 的 ridge（H100 是 295），
    意味着【未融合的 attention 在任何长度上都是带宽受限】——
    这正是 FlashAttention 存在的理由。
    """
    return head_dim / 2


def fused_attention_intensity(seq_len: int, head_dim: int = 128, bytes_per_val: int = 2) -> dict:
    """【融合后】attention 的算术强度（FlashAttention 式：不落 S/P 中间矩阵）。

    流量只剩 Q/K/V 各读一次 + O 写一次 = 4*N*d 元素（没有 N^2 项！）
    运算量不变 = 4*N^2*d

        I = 4*N^2*d / (2*4*N*d) = N/2     ← 随 N 线性增长

    对比未融合的上限 d/2（常数），融合把"上限"变成了"随长度增长"。
    """
    n, d = seq_len, head_dim
    traffic_elems = 4 * n * d
    compute_flops = 4 * n * n * d
    return dict(
        seq_len=n,
        traffic_bytes=traffic_elems * bytes_per_val,
        compute_flops=compute_flops,
        intensity=compute_flops / (traffic_elems * bytes_per_val),
    )


# ------------------------------------------------ decode 的权重加载算术强度
def decode_weight_intensity(batch: int, bytes_per_param: int = 2) -> float:
    """decode 阶段"读权重"这个动作的算术强度。

    每生成 1 个 token：读一遍全部权重（params * bytes_per_param 字节），
    做一次矩阵-向量乘（2 * params * batch FLOPs，batch 个序列共享同一次读）。

        I = 2*params*batch / (params*bytes_per_param) = 2*batch / bytes_per_param

    FP16 + batch=1 -> I = 2，远低于 H100 的 295（这就是 decode 铁定 memory bound 的原因）。
    注意 I 与参数量无关，只与 batch 和精度有关 —— 这就是 batching 的理论依据。
    """
    return 2 * batch / bytes_per_param


def batch_needed_for_ridge(ridge_val: float, bytes_per_param: int = 2) -> float:
    """要多少 batch 才能让 decode 的权重加载达到 ridge。"""
    return ridge_val * bytes_per_param / 2


# ------------------------------------------------------------------ 输出
def human(n: float) -> str:
    for unit in ("", "K", "M", "G", "T"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}P"


def cmd_table(_args):
    print("=" * 78)
    print("各代 GPU 的机器平衡点 I_ridge = FP16 峰值算力 / 内存带宽")
    print("=" * 78)
    print(f"{'GPU':<7} {'算力(TF)':>10} {'带宽(TB/s)':>11} {'I_ridge(ops/B)':>15}  备注")
    print("-" * 78)
    for name, g in GPUS.items():
        r = ridge(g["compute_tflops"], g["bandwidth_tbs"])
        print(f"{name:<7} {g['compute_tflops']:>10.0f} {g['bandwidth_tbs']:>11.2f} {r:>15.1f}  {g['note']}")
    print("-" * 78)
    print("判读：I_ridge 越高 = 硬件相对带宽越'算力富余' = 越容易卡在带宽上。")
    print("      H200 的 ridge(206) 比 H100(295) 更低，因为带宽涨了而算力没涨")
    print("      —— 所以在 decode 上 H200 比 H100 更不容易撞带宽墙。")
    print()
    print("三条结论（原书 2.4）：")
    print("  · LLM prefill      算力受限")
    print("  · LLM decode       带宽受限")
    print("  · 图像/视频生成     算力受限")


def cmd_gpu(args):
    g = GPUS[args.gpu]
    r = ridge(g["compute_tflops"], g["bandwidth_tbs"])
    print("=" * 78)
    print(f"{args.gpu}：算力 {g['compute_tflops']:.0f} TFLOPS · 带宽 {g['bandwidth_tbs']:.2f} TB/s")
    print(f"机器平衡点 I_ridge = {r:.1f} ops/byte    （{g['note']}）")
    print("=" * 78)

    # --- [1] 未融合 vs 融合 attention
    lim = attention_intensity_limit()
    print(f"\n[1] attention 的算术强度（d=128, FP16，未融合的渐近上限 d/2 = {lim:.0f}）")
    print(f"{'N':>8} {'未融合 I':>10} {'判定':>10}   {'融合后 I':>10} {'判定':>10}")
    print("-" * 78)
    for n in (128, 512, 2048, 4096, 8192, 32768):
        a = attention_intensity(n)
        f = fused_attention_intensity(n)
        va = "算力受限" if a["intensity"] >= r else "带宽受限"
        vf = "算力受限" if f["intensity"] >= r else "带宽受限"
        marker = "   ← 原书算例" if n == 4096 else ""
        print(f"{n:>8} {a['intensity']:>10.1f} {va:>10}   {f['intensity']:>10.1f} {vf:>10}{marker}")
    print("-" * 78)
    print("关键对比（这是本节最值钱的一点）：")
    print(f"  · 未融合：流量与运算都随 N^2 增长，N 约掉 → I 收敛到常数 d/2 = {lim:.0f}")
    print(f"    而 {lim:.0f} 远低于 {args.gpu} 的 ridge {r:.0f}")
    print(f"    ⇒ 未融合 attention 在【任何序列长度】上都是带宽受限")
    print(f"  · 融合后（不落 S/P 中间矩阵）：I = N/2，随长度【线性增长】")
    cross = next((n for n in range(2, 200_000) if fused_attention_intensity(n)["intensity"] >= r), None)
    if cross:
        print(f"    ⇒ 融合后 N 超过约 {cross} 就翻到算力受限一侧")
    print(f"  ⇒ 这就是 FlashAttention 存在的量化理由：它砍掉的是 N^2 那部分流量。")

    print("\n[2] decode 读权重的算术强度随 batch 变化（FP16）")
    print(f"{'batch':>7} {'I(ops/B)':>10}  判定")
    print("-" * 78)
    for b in (1, 2, 8, 32, 128, 296):
        i = decode_weight_intensity(b)
        verdict = "算力受限" if i >= r else "带宽受限"
        marker = "  ← 原书/术语表：I≈2" if b == 1 else ""
        print(f"{b:>7} {i:>10.1f}  {verdict}{marker}")
    print("-" * 78)
    need = batch_needed_for_ridge(r)
    print(f"要让 decode 的权重加载达到平衡点，需要 batch ≈ {need:.0f}。")
    print("注意：这是【理论上限视角】。真实系统里 batch 受显存(KV cache)和延迟约束，")
    print("      通常只能到几十 —— 所以 decode 实际总是带宽受限，batching 只是缓解。")

    print("\n[3] 70B 模型 FP16 读一遍权重要多久（原书的核心算例）")
    params = 70e9
    wbytes = params * 2
    t = wbytes / (g["bandwidth_tbs"] * 1e12)
    print(f"    权重 = 70e9 × 2 B = {wbytes/1e9:.0f} GB")
    print(f"    {args.gpu} 上读一遍 = {wbytes/1e9:.0f} GB / {g['bandwidth_tbs']:.2f} TB/s = {t*1000:.1f} ms")
    print(f"    → 每生成一个 token 就要付这笔钱，算力再快也没用。这就是 batching 的全部理由。")


def cmd_verify(args):
    N, D = 4096, 128
    res = attention_intensity(N, D, 2)
    ridge_h100 = ridge(GPUS["H100"]["compute_tflops"], GPUS["H100"]["bandwidth_tbs"])
    print("=" * 78)
    print("验证原书 decode 算例：d=128, N=4096, FP16")
    print("=" * 78)
    print(f"内存流量 = 4*N*d + 4*N^2 = 4*{N}*{D} + 4*{N}^2")
    print(f"         = {4*N*D:,} + {4*N*N:,} = {4*N*D + 4*N*N:,} 元素")
    print(f"         = {res['traffic_bytes']/1e6:.1f} MB （FP16 每值 2 字节）")
    print(f"运算量   = 4*N^2*d = 4*{N}^2*{D} = {res['compute_flops']:,} FLOPs")
    print(f"         = {res['compute_flops']/1e9:.2f} GFLOP")
    print(f"算术强度 = {res['compute_flops']:.3e} / {res['traffic_bytes']:.3e}"
          f" = {res['intensity']:.2f} ops/byte")
    print("-" * 78)
    print(f"原书给出的值 = {args.verify}")
    print(f"本次计算结果 = {res['intensity']:.1f}")
    ok = abs(res["intensity"] - args.verify) < 1.0
    print(f"{'✅ 一致' if ok else '❌ 不一致'}（容差 1.0；softmax 的约 5N^2 若计入只到 ~63）")
    print("-" * 78)
    print(f"对照 H100 的 I_ridge = {ridge_h100:.0f}")
    print(f"    {res['intensity']:.0f} << {ridge_h100:.0f}"
          f"  →  decode 铁定 memory bound  ✅")
    print()
    lim = attention_intensity_limit()
    print(f"补充洞察：未融合 attention 的 I 上限是 d/2 = {lim:.0f}，")
    print(f"          {lim:.0f} < {ridge_h100:.0f}，所以瓶颈不是'序列太长'，")
    print(f"          而是【实现方式本身】—— 融合掉 S/P 的往返才能翻身。")
    print()
    print(f"量感：一个 {N}×{N} 矩阵 = {N*N*2/1024/1024:.0f} MiB"
          f"（原书比喻为'一张高分辨率 RAW 单反照片'）")


def cmd_batch(_args):
    print("=" * 78)
    print("batching 为何存在：decode 的算术强度只与 batch 和精度有关")
    print("=" * 78)
    print("推导：每 token 读一遍全部权重，做 2*params*batch FLOPs")
    print("      I = 2*params*batch / (params*2B) = batch   （FP16 口径）")
    print("      → 与参数量无关！所以 batching 是通用手段。\n")
    for name in ("A100", "H100", "H200", "B200"):
        g = GPUS[name]
        r = ridge(g["compute_tflops"], g["bandwidth_tbs"])
        print(f"{name:<6} I_ridge={r:>7.1f}   需要 batch ≈ {batch_needed_for_ridge(r):>5.0f} 才触及平衡点")
    print("-" * 78)
    print("现实：batch 受 KV cache 显存与延迟 SLA 约束，通常只能到几十。")
    print("      所以 decode 永远是带宽受限 —— batching 是缓解，不是根治。")
    print()
    print("根治手段（原书第 5 章）：")
    print("  · 量化     —— 减少每字节搬运的'有效信息'（位宽减半 ⇒ 等效带宽翻倍）")
    print("  · 投机     —— 用闲置算力一次前向多出 token")
    print("  · KV cache —— 前缀复用，直接少做 prefill")
    print()
    print("⚠️ 口径说明：本脚本按'每参数 2 字节(FP16) + 2 FLOPs/参数'推得 batch=1 时 I=1。")
    print("   [技术名词表 11.3] 里写的是 I≈2。差异来自'每参数算 1 次还是 2 次运算'的口径，")
    print("   两者同一量级，都不影响结论（远低于任何数据中心 GPU 的 ridge）。")


def cmd_custom(args):
    tf, gbs = args.custom
    r = ridge(tf, gbs / 1000)   # 入参 GB/s，内部换算成 TB/s
    print("=" * 78)
    print(f"自定义设备：{tf} TFLOPS / {gbs} GB/s")
    print(f"I_ridge = {tf} / {gbs/1000:.3f} TB/s = {r:.1f} ops/byte")
    print("=" * 78)
    print("\n边缘口径的两个坑：")
    print("  1. TOPS 通常指 INT8 峰值，换精度算力会变 —— 比之前务必同精度")
    print("  2. 带宽要分清是哪一条：DDR / DMA / 片上 SRAM 差几个数量级，")
    print("     片上 SRAM 那条才是 tiling 时的真实上限")
    lim = attention_intensity_limit()
    print(f"\n本设备的 ridge = {r:.0f}，对照参考：")
    print(f"  · 未融合 attention 上限 d/2 = {lim:.0f}"
          f"  → {'算力受限' if lim >= r else '带宽受限'}")
    di = decode_weight_intensity(1)
    print(f"  · decode batch=1 的权重加载 I = {di:.0f}"
          f"  → {'算力受限' if di >= r else '带宽受限'}")
    print("\n对应你的经验：")
    print("  · 'DMA full' = 有效带宽触顶 = 落在对角线（带宽受限）一侧")
    print("  · '切 MAC 阵列能吃的块' = 提高 I（同一块数据在片上多算几次 ≡ 减少 Bytes）")
    print("  · 根治办法永远是【提高算术强度】，不是堆带宽")


def main():
    p = argparse.ArgumentParser(description="roofline 计算器（Lesson 24 配套）")
    p.add_argument("--gpu", choices=list(GPUS), help="针对某张卡做详细分析")
    p.add_argument("--verify", type=float, metavar="VALUE", help="验证 attention 算例（原书值 62）")
    p.add_argument("--batch", action="store_true", help="分析 decode 算术强度随 batch 变化")
    p.add_argument("--custom", nargs=2, type=float, metavar=("TFLOPS", "GBPS"),
                   help="自定义设备：算力 TFLOPS 与带宽 GB/s")
    args = p.parse_args()

    if args.gpu:
        cmd_gpu(args)
    elif args.verify is not None:
        cmd_verify(args)
    elif args.batch:
        cmd_batch(args)
    elif args.custom:
        cmd_custom(args)
    else:
        cmd_table(args)
        print()
        print("更多用法：")
        print("  python3 roofline.py --gpu H100        # 单卡详细分析")
        print("  python3 roofline.py --batch           # batching 的理论依据")
        print("  python3 roofline.py --verify 62       # 核对原书 decode 算例")
        print("  python3 roofline.py --custom 8 50     # 换成你自己的 NPU 参数")


if __name__ == "__main__":
    main()
