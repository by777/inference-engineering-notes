#!/usr/bin/env python3
"""为什么 prefill 算力受限、decode 带宽受限 —— 用「复用率」讲清楚

核心只有一个问题：

    同一份数据（主要是权重）搬进计算单元之后，被用来做了多少次运算？

    Prefill：权重搬 1 次 → 服务 N 个 token  → 复用 N 次   → 划算
    Decode ：权重搬 1 次 → 服务 1 个 token  → 复用 1 次   → 亏

剩下的全是这个事实的算术后果。

用法：
    python3 prefill_vs_decode.py             # 用最简线性层讲透（先看这个）
    python3 prefill_vs_decode.py --trace     # 一次请求的完整流程（看清 prefill/decode 各做什么）
    python3 prefill_vs_decode.py --attention # 澄清原书那个 62 的例子
    python3 prefill_vs_decode.py --reuse     # 复用率视角的总结（对着 此前 NPU 经验看）
"""

import argparse

FP16 = 2          # bytes per value
H100_RIDGE = 295  # ops/byte，来自 Lesson 24 第 3 章


def human_bytes(n: float) -> str:
    """按 1024 进制显示。单位用 KiB/MiB（原书也这么标，如「32 MiB」）。"""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def human_flops(n: float) -> str:
    for unit in ("", "K", "M", "G", "T"):
        if abs(n) < 1000:
            return f"{n:.2f} {unit}FLOP"
        n /= 1000
    return f"{n:.2f} PFLOP"


# ======================================================= 1. 用线性层讲透

def linear_layer(in_dim: int, out_dim: int, tokens: int):
    """算一个线性层 y = Wx 的访存与运算。

    W: [out_dim, in_dim]      权重，两种情况下都只搬一次
    x: [in_dim, tokens]       输入
    y: [out_dim, tokens]      输出

    FLOPs = 2 * out_dim * in_dim * tokens   （每个输出元素 = in_dim 次乘加 = 2*in_dim FLOPs）
    """
    w_elems = out_dim * in_dim
    x_elems = in_dim * tokens
    y_elems = out_dim * tokens
    traffic = (w_elems + x_elems + y_elems) * FP16
    flops = 2 * out_dim * in_dim * tokens
    return dict(
        w_bytes=w_elems * FP16,
        x_bytes=x_elems * FP16,
        y_bytes=y_elems * FP16,
        traffic=traffic,
        flops=flops,
        intensity=flops / traffic,
    )


def demo_linear():
    IN, OUT = 4096, 4096
    N_PREFILL = 1024

    d = linear_layer(IN, OUT, 1)              # decode：1 个 token
    p = linear_layer(IN, OUT, N_PREFILL)      # prefill：N 个 token 一起

    print("=" * 78)
    print("[1] 同一个线性层，两种用法")
    print("=" * 78)
    print(f"权重 W: [{OUT}, {IN}]，FP16。这一层占模型的权重量 = "
          f"{human_bytes(d['w_bytes'])}\n")

    rows = [
        ("搬入权重 W",           d["w_bytes"],          p["w_bytes"]),
        ("搬入输入 x / X",       d["x_bytes"],          p["x_bytes"]),
        ("写出结果 y / Y",       d["y_bytes"],          p["y_bytes"]),
        ("—— 总访存",            d["traffic"],          p["traffic"]),
        ("—— 总运算量",          d["flops"],            p["flops"]),
    ]
    print(f"{'':<22} {'Decode（1 个 token）':>24} {'Prefill（1024 个 token）':>26}")
    print("-" * 78)
    for label, dv, pv in rows:
        if "运算量" in label:
            sa, sb = human_flops(dv), human_flops(pv)
        else:
            sa, sb = human_bytes(dv), human_bytes(pv)
        print(f"{label:<22} {sa:>24} {sb:>26}")

    print("-" * 78)
    print(f"{'算术强度 I':<22} {d['intensity']:>21.1f} {p['intensity']:>26.1f}"
          f"   ops/byte")
    print(f"{'H100 平衡点':<22} {H100_RIDGE:>21} {H100_RIDGE:>26}")
    vd = "带宽受限 ❌" if d["intensity"] < H100_RIDGE else "算力受限"
    vp = "带宽受限" if p["intensity"] < H100_RIDGE else "算力受限 ✅"
    print(f"{'判定':<22} {vd:>22} {vp:>28}")
    print("=" * 78)

    print("\n★ 请盯着这两行看：")
    print(f"    访存：   {human_bytes(d['traffic'])}  →  {human_bytes(p['traffic'])}"
          f"      （只涨了 {p['traffic']/d['traffic']:.1f} 倍）")
    print(f"    运算：   {human_flops(d['flops'])}  →  {human_flops(p['flops'])}"
          f"    （涨了 {p['flops']/d['flops']:.0f} 倍）")
    print()
    print("    权重那一份【完全没变】，但被 1024 个 token 复用了。")
    print("    分子涨 1024 倍、分母几乎不动 ⇒ 算术强度从 1.0 涨到 683。")
    print()
    print("    这就是 prefill 和 decode 的全部差别。")
    print("    ※ decode 时权重每个 token 都要重新搬一遍，搬完只用 1 次 ⇒ 血亏。")
    print("    ※ 70B FP16 每 token 搬 140GB、H100 上光读 42ms，就是这么来的。")
    print()
    print("    所以 batching 有用：batch=32 就是让同一份权重服务 32 个序列，")
    print("    复用率从 1 提到 32，把 I 从 1 拉到 32（仍远低于 295，但好 32 倍）。")


# ========================================================== 2. attention 澄清

def attn_decode(seq: int, d: int):
    """【真实的 decode 形状】Q 只有 1 行（当前正在生成的那个 token）。

        Q:[1,d]  K:[N,d]  V:[N,d]   →   S=QK^T:[1,N]  P:[1,N]  O=PV:[1,d]
    """
    read = d + seq * d + (1 * seq) + (1 * seq) + (seq * d)
    write = (1 * seq) + (1 * seq) + d
    flops = 2 * (1 * seq * d) + 5 * (1 * seq) + 2 * (1 * d * seq)
    traffic = (read + write) * FP16
    return dict(read=read, write=write, traffic=traffic, flops=flops,
                intensity=flops / traffic)


def attn_prefill(seq: int, d: int):
    """【原书例子的形状】Q 有 N 行（等价于把整个序列一起算，即 prefill）。

        Q:[N,d]  K:[N,d]  V:[N,d]   →   S=QK^T:[N,N]  P:[N,N]  O=PV:[N,d]
    """
    n = seq
    read = (n * d) + (n * d) + (n * n) + (n * n) + (n * d)
    write = (n * n) + (n * n) + (n * d)
    flops = 2 * (n * n * d) + 5 * (n * n) + 2 * (n * d * n)
    traffic = (read + write) * FP16
    return dict(read=read, write=write, traffic=traffic, flops=flops,
                intensity=flops / traffic)


def demo_attention():
    N, D = 4096, 128
    dc = attn_decode(N, D)
    pf = attn_prefill(N, D)

    print("=" * 78)
    print("[2] 澄清：原书那个「62」到底是什么形状")
    print("=" * 78)
    print("原书写：『consider a decode step ... on a sequence of 4096 tokens』")
    print("      然后给出 Q, K, V 都是 N×d = 4096×128\n")
    print("问题：decode 时 Q 只有【1 行】（就当前这一个 token），")
    print("      K/V 才是 N 行（从 KV cache 读全部历史）。")
    print("      所以把 Q 写成 N×d，其实是【prefill 的形状】。\n")

    print(f"{'':<26} {'真实 decode':>16} {'原书例子的形状':>18}")
    print(f"{'':<26} {'Q:[1,d]':>16} {'Q:[N,d]（=prefill）':>18}")
    print("-" * 78)
    print(f"{'S = QK^T 的形状':<26} {'[1, 4096]':>16} {'[4096, 4096]':>18}")
    print(f"{'S 的元素数':<26} {1*N:>16,} {N*N:>18,}")
    print(f"{'访存（元素）':<26} {dc['read']+dc['write']:>16,} "
          f"{pf['read']+pf['write']:>18,}")
    print(f"{'访存（字节）':<26} {human_bytes(dc['traffic']):>16} "
          f"{human_bytes(pf['traffic']):>18}")
    print(f"{'运算量':<26} {human_flops(dc['flops']):>16} "
          f"{human_flops(pf['flops']):>18}")
    print(f"{'算术强度 I':<26} {dc['intensity']:>16.1f} {pf['intensity']:>18.1f}")
    print(f"{'H100 平衡点':<26} {H100_RIDGE:>16} {H100_RIDGE:>18}")
    print("-" * 78)
    vd = "带宽受限" if dc["intensity"] < H100_RIDGE else "算力受限"
    vp = "带宽受限" if pf["intensity"] < H100_RIDGE else "算力受限"
    print(f"{'判定':<26} {vd:>16} {vp:>18}")
    print("=" * 78)
    print("\n发现两件事：")
    print(f"  ① 原书那个 62，用 Q:[N,d] 的形状才能算出来（真实 decode 只有 "
          f"{dc['intensity']:.1f}）")
    print(f"  ② 但两者【都】远小于 295，所以『decode 是带宽受限』的结论没错——")
    print(f"     真实 decode 甚至更极端（{dc['intensity']:.1f} vs 62，差 "
          f"{pf['intensity']/dc['intensity']:.0f} 倍）")
    print()
    print("为什么会不一致？因为 decode 把 QK^T 从")
    print("     【大矩阵乘】[N,d]×[d,N] → [N,N]")
    print("  变成了")
    print("     【向量-矩阵乘】[1,d]×[d,N] → [1,N]")
    print("    —— 权重（这里是 K）搬进来只用了 1 行 Q 去乘，复用率极低。")
    print()
    print("※ 顺带说明：原书用 N×N 的形状也是有意的——它想说明")
    print("   【即使按最有利于算力的形状摆开，未融合的 attention 也只有 62】，")
    print("   这恰恰是 FlashAttention 要解决的问题。只是它标成 'decode step'")
    print("   会让读者以为这就是 decode 的真实形状，容易卡住。")
    print()
    print("   注意：prefill 整体【仍然】是算力受限的——因为 prefill 的大头")
    print("   是那些线性层（QKV projection、FFN），它们是高复用的大矩阵乘")
    print("   （见 [1] 里的 683）。attention 只是 prefill 里偏低的一块。")


# =============================================================== 3. 复用率

def demo_reuse():
    IN, OUT = 4096, 4096
    print("=" * 78)
    print("[3] 复用率视角（对着 此前 NPU 的经验看）")
    print("=" * 78)
    print("把 [1] 的表格换个写法：固定权重只搬一次，看复用率怎么变\n")
    print(f"{'tokens':>8} {'总访存':>12} {'总运算':>14} {'算术强度':>12}  判定")
    print("-" * 78)
    for t in (1, 2, 8, 32, 128, 1024):
        r = linear_layer(IN, OUT, t)
        v = "带宽受限" if r["intensity"] < H100_RIDGE else "算力受限"
        mark = "  ← decode" if t == 1 else ("  ← prefill 的典型形态" if t == 1024 else "")
        print(f"{t:>8} {human_bytes(r['traffic']):>12} {human_flops(r['flops']):>14}"
              f" {r['intensity']:>9.1f}  {v}{mark}")
    print("-" * 78)
    print("\n这张表就是「同一个操作，为什么换个用法瓶颈就换了」的全部答案：")
    print()
    print("  复用率 = 一份数据搬进来，参与了多少次运算")
    print("         = prefill 的 N 个 token / decode 的 1 个 token")
    print()
    print("★ 这和你此前 NPU 做的事是同一个东西：")
    print("    · 「切 MAC 阵列能吃的块、索引复用」 = 让一块数据在片上多算几次")
    print("      → 就是在提高复用率 → 就是在提高算术强度 I")
    print("    · 「DMA full」= 带宽触顶 = 落在 I < I_ridge 那一侧")
    print("    · 「pingpong 双缓冲」= 搬运和计算重叠，掩盖访存")
    print()
    print("  所以 roofline 不是新知识，是给你已经在做的事配了一套坐标系。")
    print()
    print("★ decode 的三个解法，本质都在提高复用率或减少搬的东西：")
    print("    · batching      → 让同一份权重服务多个序列（复用率 ×batch）")
    print("    · 量化          → 权重从 2 字节变 1 字节（分母减半）")
    print("    · 投机解码      → 一次前向多出几个 token（复用率按接受数提升）")
    print("    · KV cache 复用 → 直接少搬历史（减少要搬的字节）")


# ============================================== 4. 一次请求的完整流程

def demo_trace():
    """把一次请求拆成「prefill 一次 + decode M 次」，看清每个阶段各做什么。"""
    MODEL_B = 7            # 模型参数量（十亿）
    N_IN = 64              # 输入序列 token 数
    N_OUT = 32             # 生成 token 数
    BANDWIDTH = 3.35e12    # H100 带宽 B/s

    w_bytes = MODEL_B * 1e9 * FP16
    read_ms = w_bytes / BANDWIDTH * 1000

    print("=" * 78)
    print("[4] 一次请求的完整流程（看清 prefill 和 decode 各做什么）")
    print("=" * 78)
    print(f"设定：{MODEL_B}B 参数模型（FP16 权重 = {human_bytes(w_bytes)}）")
    print(f"      H100 带宽 3.35 TB/s → 【读一遍权重】= {read_ms:.2f} ms")
    print(f"      输入 {N_IN} 个 token，要生成 {N_OUT} 个 token\n")

    print("阶段 0：tokenize（不需要神经网络）")
    print(f"    '文本' → [{N_IN} 个 token id]\n")

    print("─" * 78)
    print("【Prefill】把整个输入序列一次并行处理")
    print("─" * 78)
    print(f"    喂入    ：{N_IN} 个 token（一次前向）")
    print(f"    读取    ：全部权重（{human_bytes(w_bytes)}）+ 还没有历史 KV")
    print(f"    计算    ：为这 {N_IN} 个 token 各算 attention，【写入】KV cache")
    print(f"    产出    ：logits → 采样出【第 1 个】输出 token")
    print(f"    权重读取：1 次 = {read_ms:.2f} ms （但服务了 {N_IN} 个 token，划算）")
    print(f"    瓶颈    ：算力受限 → 决定 TTFT")

    print("\n" + "─" * 78)
    print(f"【Decode】逐 token 生成剩下 {N_OUT - 1} 个 token")
    print("─" * 78)
    print(f"{'第几个':>8} {'喂入':>10} {'读取':>22} {'KV cache':>12} {'权重读取':>12}")
    print("-" * 78)
    total_reads = 1
    for i in range(1, min(N_OUT, 6)):
        kv_len = N_IN + i
        print(f"{i:>8} {'1 个 token':>12} {'权重 + KV cache':>22} "
              f"{kv_len:>10} 位 {read_ms:>10.2f} ms")
        total_reads += 1
    if N_OUT > 6:
        print(f"{'...':>8} {'...':>12} {'...':>22} {'...':>12} {'...':>12}")
        total_reads = N_OUT          # prefill 1 次 + decode (N_OUT-1) 次 ≈ N_OUT
    print("-" * 78)
    print(f"    每个 decode 步只喂【1 个】token，但都要【重新读一遍全部权重】")
    print(f"    产出    ：每步 1 个 token（直到吐 <eos> 或撞上限）")
    print(f"    瓶颈    ：带宽受限 → 决定 TPS")

    print("\n" + "=" * 78)
    print("把账算一下")
    print("=" * 78)
    ideal = read_ms * total_reads
    print(f"    权重被读的总次数 = 1（prefill）+ {total_reads - 1}（decode）"
          f" = {total_reads} 次")
    print(f"    光读权重的总耗时 ≈ {total_reads} × {read_ms:.2f} ms = {ideal:.0f} ms")
    print()
    print(f"    其中 prefill 那 1 次服务了 {N_IN} 个 token（复用率 {N_IN}）")
    print(f"    而每次 decode 只服务 1 个 token（复用率 1）← 这就是瓶颈所在")
    print()
    print("★ 两个纠正认知的点：")
    print(f"  ① prefill 只跑【1 次】，decode 跑【{total_reads - 1} 次】")
    print(f"     → 长输出时，绝大部分时间花在 decode 上")
    print(f"  ② 它们【跑的是完全相同的权重】——不是两个模型，通常也不是两段代码")
    print(f"     → 实现上往往只是张量形状不同：{N_IN}×d（一次喂 N 个）vs 1×d（一次喂 1 个）")


def main():
    p = argparse.ArgumentParser(description="prefill vs decode：复用率视角")
    p.add_argument("--attention", action="store_true", help="澄清 attention 的 62")
    p.add_argument("--reuse", action="store_true", help="复用率总结")
    p.add_argument("--trace", action="store_true", help="一次请求的完整流程")
    args = p.parse_args()

    if args.attention:
        demo_attention()
    elif args.reuse:
        demo_reuse()
    elif args.trace:
        demo_trace()
    else:
        demo_linear()
        print()
        print("继续看：")
        print("  python3 prefill_vs_decode.py --trace       # 一次请求的完整流程")
        print("  python3 prefill_vs_decode.py --attention   # 澄清原书的 62")
        print("  python3 prefill_vs_decode.py --reuse       # 复用率总结")


if __name__ == "__main__":
    main()
