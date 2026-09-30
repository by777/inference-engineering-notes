# 02_real_model_ptq.py
# Lesson 21 实验 2: 真实模型中间值分布 + 量化误差模拟
#
# 目标:
#   1. 读 onnx_layer_dump 工具导出的 fp32 中间值（runtime_dump/*.bin）
#   2. 分析各层激活的分布形状（均匀/长尾）→ 用 near0 比例 + 峰度 kurtosis
#   3. 模拟 PTQ 量化（per-tensor / per-channel）对比 SNR, 找出量化敏感层
#
# 前置: model_trans 项目已 dump 出 <out>/runtime_dump/ 下的 fp32 中间值
# 运行: python 02_real_model_ptq.py --model_dir <dump输出目录>
#

import argparse
import json
import numpy as np
from pathlib import Path


def quantize_symmetric(x, bits=8):
    """对称量化, 返回 (q, scale)"""
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(x)) / qmax
    q = np.clip(np.round(x / scale), -qmax - 1, qmax).astype(np.int16)
    return q, scale


def snr_db(x, x_hat):
    err = x - x_hat
    return 20 * np.log10(np.linalg.norm(x) / (np.linalg.norm(err) + 1e-12))


def analyze_layer(name, x):
    """分析单个张量: 分布特征 + per-tensor/per-channel 量化 SNR"""
    x = x.astype(np.float32)
    flat = x.flatten()

    # ---- 分布特征 ----
    xmax = np.max(np.abs(flat))
    # near0: 接近 0 的比例（相对 max 的 1% 以内）→ 长尾/稀疏度的指标
    n_near_zero = np.mean(np.abs(flat) < 0.01 * xmax) if xmax > 0 else 1.0
    # kurtosis: 峰度, 正态=3, 长尾>>3
    kurt = ((flat - flat.mean()) ** 4).mean() / (flat.var() + 1e-12) ** 2

    # ---- per-tensor 量化 ----
    q, s = quantize_symmetric(x)
    x_hat = q.astype(np.float32) * s
    snr_pt = snr_db(x, x_hat)

    # ---- per-channel 量化（只对 4D 激活按 C、2D 权重按 C_out 切）----
    # 1D/小 slice 不做 (per-channel 失去意义, 会虚高)
    snr_pc = float("nan")
    if x.ndim == 4 and x.shape[1] > 1:  # [N,C,H,W] → per-C
        axes = (0, 2, 3)
        scales = np.max(np.abs(x), axis=axes) / 127.0  # [C]
        q_pc = np.clip(np.round(x / scales[None, :, None, None]), -128, 127).astype(
            np.int16
        )
        x_hat_pc = q_pc.astype(np.float32) * scales[None, :, None, None]
        snr_pc = snr_db(x, x_hat_pc)
    elif x.ndim == 2 and x.shape[0] > 8:  # [C_out,C_in] → per-C_out
        scales = np.max(np.abs(x), axis=1) / 127.0  # [C_out]
        q_pc = np.clip(np.round(x / scales[:, None]), -128, 127).astype(np.int16)
        x_hat_pc = q_pc.astype(np.float32) * scales[:, None]
        snr_pc = snr_db(x, x_hat_pc)

    # ---- 敏感度标记 ----
    flag = ""
    if snr_pt < 25:
        flag += "低SNR "
    if n_near_zero > 0.8:
        flag += "长尾"
    if flag:
        flag = "  [" + flag.strip() + "]"

    print(
        f"  {name:36s} shape={str(x.shape):18s} near0={n_near_zero:5.1%} "
        f"kurt={kurt:7.1f} snr_pt={snr_pt:6.1f}dB snr_pc={snr_pc:6.1f}dB{flag}"
    )
    return {
        "name": name,
        "snr_pt": float(snr_pt),
        "snr_pc": float(snr_pc),
        "near0": float(n_near_zero),
        "kurt": float(kurt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir", required=True, help="dump 输出目录(含 runtime_dump/)"
    )
    ap.add_argument("--max_layers", type=int, default=60, help="最多分析多少层")
    args = ap.parse_args()

    dump_dir = Path(args.model_dir) / "runtime_dump"
    if not dump_dir.exists():
        print(f"找不到 {dump_dir}，请检查 --model_dir")
        return

    metas = sorted(dump_dir.glob("*.meta.json"))
    print(f"共 {len(metas)} 个张量\n")

    results = []
    count = 0
    for meta_path in metas:
        meta = json.loads(meta_path.read_text())
        # 只分析中间激活（跳过 graph_input/graph_output/Constant 权重）
        if meta.get("role") != "intermediate":
            continue
        # 只分析 float32（Shape 算子是 int64, 量化分析无意义）
        if meta.get("dtype") != "float32":
            continue
        bin_path = dump_dir / meta["bin_file"]
        if not bin_path.exists():
            continue
        x = np.fromfile(bin_path, dtype=np.float32)
        # shape 容错: 用实际数据大小 reshape（动态 shape 元信息不准）
        shape = meta.get("shape") or []
        n_elems = int(np.prod(shape)) if shape else -1
        if n_elems == x.size and n_elems > 0:
            x = x.reshape(shape)
        elif x.size == 1:
            x = x.reshape(1)
        else:
            x = x.reshape(-1)
        # 过滤: 标量/极小张量(<=4 元素)、全零张量
        if x.size <= 4 or np.max(np.abs(x)) == 0:
            continue
        results.append(analyze_layer(meta.get("name", "?"), x))
        count += 1
        if count >= args.max_layers:
            break

    # ---- 汇总: 量化最敏感的层 ----
    print("\n=== 量化最敏感的层 (snr_pt < 25dB) ===")
    sensitive = [r for r in results if r["snr_pt"] < 25]
    if sensitive:
        for r in sorted(sensitive, key=lambda r: r["snr_pt"])[:10]:
            print(
                f"  {r['name']:36s} snr_pt={r['snr_pt']:.1f}dB near0={r['near0']:.1%} kurt={r['kurt']:.1f}"
            )
    else:
        print("  无 —— 所有层 snr_pt >= 25dB，整体可量化")

    # per-channel 增益最大的层
    print("\n=== per-channel 增益最大的层 (snr_pc - snr_pt > 5dB) ===")
    gains = [r for r in results if r["snr_pc"] - r["snr_pt"] > 5]
    if gains:
        for r in sorted(gains, key=lambda r: r["snr_pc"] - r["snr_pt"], reverse=True)[
            :10
        ]:
            print(
                f"  {r['name']:36s} pt={r['snr_pt']:.1f}dB -> pc={r['snr_pc']:.1f}dB  (+{r['snr_pc']-r['snr_pt']:.1f})"
            )
    else:
        print("  无 —— per-channel 增益不明显（通道尺度差异小）")

    out = dump_dir.parent / "quant_analysis.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n分析结果已保存: {out}")


if __name__ == "__main__":
    main()
