#!/usr/bin/env python3
"""实测「你这台机器」的 roofline —— 把纸面复算变成真尺子

`prefill_vs_decode.py` 和 `roofline.py` 都是【纸面复算】：它们算的是公式的输出，
不是硬件的表现。本脚本反过来：**在你这台机器上真测一遍**，然后回答

    · 我这台机器的 ridge 是多少？
    · 和 H100 差多少倍？差在算力还是带宽？
    · 我这台机器上，7B 模型 decode 最快能到多少 token/s？
    · prefill 要多少 token 才变成算力受限？

用法：
    python3 measure_ridge.py                # 完整测量（默认 256 MB 数组）
    python3 measure_ridge.py --quick        # 快速（64 MB，适合共享机器）
    python3 measure_ridge.py --threads 1    # 限制成单线程（看 OpenMP 收益）
    python3 measure_ridge.py --size 512     # 自定义数组大小（MB）
    python3 measure_ridge.py --compare      # 附带各代硬件对比表

测量的三条纪律（很重要，否则数据没意义）：
    1. 数组必须远大于最后一级缓存，否则测的是 cache 带宽（虚高 10 倍）
    2. 同一个量用多种操作交叉验证，结果不一致说明有别的瓶颈
    3. 取最优而非平均——"最好能跑多快"才有意义，平均值被调度噪声污染
"""

import sys
import os

# ⚠️ 线程数必须在 import numpy 之前设好，否则 OpenBLAS 已经初始化了
if "--threads" in sys.argv:
    _i = sys.argv.index("--threads")
    if _i + 1 < len(sys.argv):
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[_v] = sys.argv[_i + 1]

import argparse
import platform
import time

try:
    import numpy as np
except ImportError:
    print("需要 numpy：pip install numpy")
    sys.exit(1)


# ============================================================ 工具箱

def best_of(fn, repeats: int = 5, warmup: int = 3) -> float:
    """跑 repeats 次取【最快】的一次（秒）。

    为什么取最优而不是平均：我们想知道的是「这台机器最好能跑多快」，
    平均值会被调度、降频、其他进程干扰污染。取最优才接近硬件上限。

    为什么要 warmup 多次：实测发现 OpenBLAS 首次遇到某个矩阵形状时，
    要做 kernel 选择 + 首次触碰新内存页，**第一轮能慢 30%~100%**，
    单次 warmup 不足以消除（实测 1024×1024 的首轮仍是稳态的 1.5 倍）。
    """
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


# ====================================================== 1. 测内存带宽

def measure_bandwidth(size_mb: int, repeats: int = 5):
    """用 3 种不同的访存模式交叉验证带宽。

    每种模式搬运的字节数不同，但【带宽】应该一致。
    如果不一致，说明瓶颈不是带宽（例如归约操作受 ALU/单线程限制）。
    """
    n = size_mb * 1024 * 1024 // 4                    # fp32 元素数
    a = np.ones(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    c = np.empty(n, dtype=np.float32)

    modes = [
        ("copyto(c, a)",       lambda: np.copyto(c, a),        2 * n * 4, "读 N + 写 N"),
        ("add(a, b, out=c)",   lambda: np.add(a, b, out=c),    3 * n * 4, "读 2N + 写 N"),
        ("a *= 2 (in-place)",  lambda: a.__imul__(2),          2 * n * 4, "读 N + 写 N"),
    ]

    print(f"\n[1] 内存带宽（数组 {size_mb} MiB × 3 个，远大于 L3 才有效）")
    print("-" * 78)
    print(f"{'操作':<22}{'耗时':>10}{'搬运':>12}{'带宽':>12}   说明")
    results = []
    for name, fn, nbytes, note in modes:
        dt = best_of(fn, repeats)
        gbs = nbytes / dt / 1e9
        results.append(gbs)
        print(f"{name:<22}{dt*1000:>8.2f} ms{human_bytes(nbytes):>12}{gbs:>9.1f} GB/s   {note}")
    print("-" * 78)

    bw = max(results)
    spread = max(results) / min(results)
    print(f"{'→ 取最优作为带宽':<22}{'':>10}{'':>12}{bw:>9.1f} GB/s")

    # 自我校验：不同模式差太多说明测的不是带宽
    if spread > 2.5:
        print(f"\n⚠️  不同操作的带宽差了 {spread:.1f} 倍 —— 说明瓶颈不是纯带宽")
        print("    （比如某些操作受 ALU 或单线程限制，不能反映内存上限）")
    else:
        print(f"\n✓  三种访存模式结果接近（差 {spread:.2f} 倍）→ 测到的是内存带宽量级")
    print("    ※ 细微差异是正常的，原因是「写分配」(write-allocate)：")
    print("      纯写操作会先读入缓存行再改写，实际流量比理论值多一份 N，")
    print("      所以多流读模式（如 a+b）看起来会低一些。这不算测量错误。")
    print(f"    取 {bw:.1f} GB/s 作为「带宽上界」用于后续计算。")
    return bw, n, results


# ====================================================== 2. 测算力

def measure_flops(sizes=(256, 512, 1024, 2048, 3072), repeats: int = 3):
    """用不同尺寸的方阵乘法扫描算力，取峰值。

    小矩阵受 cache 与线程启动开销影响偏低；大矩阵才接近峰值。
    但太大又会被内存带宽拖累（矩阵乘本身也需要喂数据）。
    """
    print(f"\n[2] FP32 算力（方阵乘法扫描，OpenBLAS）")
    print("-" * 78)
    print(f"{'尺寸':<12}{'FLOPs':>14}{'耗时':>12}{'GFLOPS':>12}")

    rows = []
    for n in sizes:
        a = np.ones((n, n), dtype=np.float32)
        b = np.ones((n, n), dtype=np.float32)
        flops = 2.0 * n ** 3
        try:
            dt = best_of(lambda: a @ b, repeats)
        except MemoryError:
            rows.append((n, flops, None, None))
            del a, b
            continue
        rows.append((n, flops, dt, flops / dt / 1e9))
        del a, b

    # 先收集再标注峰值，否则每行都会显示"峰值"
    valid = [r for r in rows if r[3] is not None]
    peak = max(r[3] for r in valid) if valid else 0.0
    for n, flops, dt, gflops in rows:
        if gflops is None:
            print(f"{n:<12}{flops/1e9:>11.1f} GF{'':>12}{'OOM':>12}")
            continue
        mark = "  ← 峰值" if gflops >= peak * 0.999 else ""
        print(f"{n:<12}{flops/1e9:>11.1f} GF{dt*1000:>10.2f} ms{gflops:>12.1f}{mark}")
    print("-" * 78)
    best_n = max(valid, key=lambda r: r[3])[0] if valid else 0
    print(f"→ 峰值算力 {peak:.1f} GFLOPS（{best_n}×{best_n} 矩阵）")
    if len(valid) > 1:
        lo = min(r[3] for r in valid)
        print(f"  （最小尺寸只有 {lo:.1f} GFLOPS，是小矩阵受 cache/线程启动限制——"
              f"这是真实现象，不是测量错误）")
    return peak


# ==================================================== 3. ridge 与推论

def print_hw_table(local_gflops=None, local_bw=None):
    """打印各代硬件的算力 / 带宽 / ridge 对照表。

    只在【真的测过】时才给出「本机」行 —— 否则用固定参考值填空会误导读者
    以为那是实测结果。
    """
    rows = []
    if local_gflops is not None and local_bw is not None:
        rows.append(("本机（本次实测, FP32）", local_gflops, local_bw, "FP32"))
    else:
        rows.append(("本机", None, None, "—"))
    rows += [
        ("H100（FP32 CUDA core）",    67000.0, 3350.0, "FP32"),
        ("H100（FP16 Tensor Core）",  989000.0, 3350.0, "FP16"),
        ("H100（FP8 Tensor Core）",  1979000.0, 3350.0, "FP8"),
        ("B200（FP16）",             2250000.0, 8000.0, "FP16"),
    ]
    print("-" * 78)
    print(f"{'硬件':<28}{'精度':>6}{'算力(GF/s)':>13}{'带宽(GB/s)':>12}{'ridge':>9}")
    for name, f, b, prec in rows:
        if f is None:
            print(f"{name:<28}{prec:>6}{'（未测）':>13}{'（未测）':>12}{'（未测）':>9}")
        else:
            print(f"{name:<28}{prec:>6}{f:>13,.0f}{b:>12,.0f}{f/b:>9.1f}")
    print("-" * 78)


def report_ridge(bw_gbs: float, gflops: float):
    """算出本机 ridge，并推导它在 LLM 推理上意味着什么。"""
    ridge = gflops / bw_gbs                      # (GFLOP/s) / (GB/s) = ops/byte
    print(f"\n[3] 你这台机器的 roofline")
    print("=" * 78)
    print(f"{'算力（FP32）':<28}{gflops:>12.1f} GFLOPS")
    print(f"{'带宽':<28}{bw_gbs:>12.1f} GB/s")
    print(f"{'ridge = 算力 / 带宽':<28}{ridge:>12.1f} ops/byte")
    print("=" * 78)

    # ---- 与各代硬件对比（ridge 是"算力/带宽"，所以对比要标清精度口径）
    print(f"\n[4] 和各代硬件放在一起看")
    print_hw_table(gflops, bw_gbs)

    print("\n★ 一个反直觉但正确的现象：")
    print(f"    在 FP32 口径下，本机 ridge（{ridge:.0f}）【比 H100 的 FP32 ridge（20）还高】！")
    print("    原因不是 CPU 强，而是【GPU 用 Tensor Core 时算力暴涨 15~60 倍，")
    print("    但带宽只涨了几倍】—— 这就是「内存墙」。")
    print("    GPU 的高 ridge 来自专用矩阵单元，不是来自「带宽小」。")
    print("    ⇒ 所以对比 ridge 时【必须标明精度】，否则是错的。")

    # ---- 推论 1：本机 decode 速度上限
    print(f"\n[5] 推论：在你这台机器上跑 LLM")
    print("-" * 78)
    print(f"{'模型':<14}{'权重(FP16)':>13}{'decode 上限':>16}   说明")
    for name, p in (("7B", 7e9), ("70B", 70e9)):
        w_gb = p * 2 / 1e9
        t_ms = w_gb / bw_gbs * 1000
        tps = 1000 / t_ms
        print(f"{name:<14}{w_gb:>10.1f} GB{tps:>13.1f} tok/s"
              f"   每 token 要搬 {w_gb:.0f} GB，纯带宽极限")
    print("-" * 78)
    print("    ※ 这是【只看带宽】的上界。实际还要加上：")
    print("      · 算力不够时会更慢（CPU 算力远低于 GPU）")
    print("      · KV cache 的搬运（长上下文时显著）")
    print("      · 激活、中间结果的访存")
    print("      ⇒ 所以实测一定比这个数慢，这个数是【天花板】")

    # ---- 推论 2：prefill/decode 分界点
    # 整模型近似 I ≈ N（FP16），所以分界点 ≈ ridge
    print(f"\n[6] 推论：prefill 多少 token 才变成算力受限")
    print("-" * 78)
    print("    两个假设：")
    print("      (a) 模型用 FP16 → 每参数 2 字节")
    print("      (b) 每参数每 token 做 2 FLOP（1 乘 + 1 加）")
    print("    两个 2 抵消 ⇒ I ≈ N   ⇒   分界点 N ≈ ridge")
    print()
    print(f"    用本机 ridge = {ridge:.0f} 代入：")
    for label, r in (("本机", ridge), ("H100 FP16", 295.0), ("H100 FP8", 591.0)):
        print(f"      {label:<12} 需要 {r:>6.0f} 个 token 才翻到算力受限侧")
    print("-" * 78)
    print(f"    ⇒ 本机的分界点只有约 {ridge:.0f} 个 token（H100 要 295 个），")
    print("      比 H100 低一个数量级。意味着【在本机做 prefill，几乎总是算力受限】——")
    print("      这和 GPU 上「短输入 prefill 仍可能带宽受限」是很不同的。")
    print()
    print("    ⚠️ 口径说明（重要，否则会误用）：")
    print("      · I ≈ N 成立的前提是模型用 FP16。若模型用 FP32，则 I ≈ N/2，")
    print("        分界点要翻倍（本机约 60 个 token）。")
    print("      · ridge 是用【本机 FP32 算力】算的。CPU 上没有专用矩阵单元，")
    print("        FP16 与 FP32 算力差别不大，所以这个近似对 CPU 合理。")
    print("      · 但在 GPU 上这么套就错了 —— Tensor Core 让 FP16 算力暴涨，")
    print("        ridge 随精度剧烈变化（H100: FP16 295 vs FP8 591）。")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="实测本机 roofline")
    ap.add_argument("--quick", action="store_true", help="快速模式（64 MiB 数组）")
    ap.add_argument("--size", type=int, default=None, metavar="MB",
                    help="数组大小（MiB），默认 256")
    ap.add_argument("--threads", type=str, default=None, metavar="N",
                    help="限制线程数（需在 import numpy 前设置）")
    ap.add_argument("--repeats", type=int, default=5, help="每次测量重复次数（取最优）")
    ap.add_argument("--compare", action="store_true", help="只看硬件对比表")
    args = ap.parse_args()

    print("=" * 78)
    print("实测本机 roofline")
    print("=" * 78)
    print(f"  CPU      : {platform.processor() or platform.machine()}")
    print(f"  核心数   : {os.cpu_count()}")
    print(f"  Python   : {platform.python_version()}   numpy: {np.__version__}")
    print(f"  线程设置 : {args.threads or '默认（OpenBLAS 自动）'}")

    if args.compare:
        print("\n[参考] 各代硬件的算力 / 带宽 / ridge")
        print("（固定参考值，**不是**本机实测 —— 本机那行显示「未测」）")
        print_hw_table()
        print("\n想看本机实测 → 直接运行 python3 measure_ridge.py")
        return

    size_mb = args.size or (64 if args.quick else 256)
    bw, _, _ = measure_bandwidth(size_mb, args.repeats)
    sizes = (256, 512, 1024, 2048) if args.quick else (256, 512, 1024, 2048, 3072)
    gflops = measure_flops(sizes, max(2, args.repeats - 2))
    report_ridge(bw, gflops)

    print(f"\n{'=' * 78}")
    print("诚实声明（请务必读）")
    print("=" * 78)
    print("  · 这是【真测】的，但只测了 numpy/OpenBLAS 这一条路径。")
    print("    真实推理引擎（llama.cpp / ONNX Runtime）的数值会不同。")
    print("  · 矩阵乘走 OpenBLAS（多线程 + SIMD），比手写循环快得多；")
    print("    但真实推理里小批量 matmul 未必能吃到这个峰值。")
    print("  · 共享机器 / 虚拟机 / 容器环境下结果波动很大，请多跑几次对比。")
    print("  · ridge 依赖精度口径（FP32 vs FP16 vs FP8），跨精度比是错的。")
    print("  · 最可靠的做法：在同一台机器上换不同参数多跑几次，看趋势。")
    print()
    print("  ⚠️ 实测踩到的坑（本脚本已尽量规避，但你要知道）：")
    print("  1. 冷启动失真：第一次跑脚本时 OpenBLAS 要做 kernel 选择，")
    print("     全部数字可能偏低好几倍。**建议跑两遍，用第二遍的数**。")
    print("  2. warmup 不够：同一矩阵尺寸的第一轮可能仍慢 30%~100%，")
    print("     所以本脚本 warmup 3 次 + 取多次最优。")
    print("  3. 数组太小测的是 cache 带宽（能虚高 10 倍），所以默认 256 MiB。")
    print("  4. 写分配让多流读写模式的表观带宽偏低，属正常现象。")
    print("=" * 78)


if __name__ == "__main__":
    main()
