# 04_integer_inference.py
# Lesson 21: 实验4：真实模型的整数推理闭环
# 链路: 真实中间tensor + 真实权重
#   → int8量化 → 整数 depthwise conv (int32累加) → requantize(除以输出scale) → 对比浮点真值
# 回答: ① 为什么必须 int32 累加
#        ② requantize 为什么"除以输出 scale" (和 scale_w·scale_x 什么关系)
#        ③ 全程整数(输入int8→输出int8) vs 浮点推理, 差多少
#
# ⚠ 正确链路 (从 fsmn_out/layers/layer_0033_Conv.json 读到, 不能靠名字猜!):
#   Concat(cache拼接) → Slice_1 [1,128,13,1] → conv_left [1,128,4,1]
#   conv_left 吃的是 13 帧历史窗口 (流式 cache 拼接), pads=[0,0,0,0] 无padding
#   Transpose 是喂给 conv_right 的, 不是 conv_left!

# ###################### 代码目标 #######################
# 输入 int8 → 权重 int8 → int32 累加 → requantize 回 int8
# #######################################################

import json
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fsmn_out"
RD = OUT / "runtime_dump"
WD = OUT / "weights"


def load_tensor(path):
    """读 dump 的 .bin + .meta.json, 还原成 float32 ndarray"""
    meta = json.load(open(path.with_suffix(".meta.json"), "r"))
    t = np.fromfile(path, dtype=np.float32).reshape(meta["shape"])
    return t, meta


def scale_to_fixed(scale, bits, min_mult=0, max_mult=None):
    """扫描法: 浮点 scale → (multiplier, shift), 复用实验3
    核心: 找 (mult, shift) 使 mult/2^shift ≈ scale, 且 mult 不超位宽"""
    if max_mult is None:
        max_mult = (1 << bits) - 1
    best = None
    for shift in range(0, 2 * bits + 1):
        mult = round(scale * (1 << shift))
        if mult < min_mult or mult > max_mult:
            continue
        scale_fixed = mult / (1 << shift)
        err = abs(scale_fixed - scale) / scale
        if best is None or err < best[0]:
            best = (err, mult, shift)
    return best[1], best[2]


def depthwise_conv1d(x, w, pl, pr):
    """1D depthwise 卷积 (互相关, ONNX Conv 不翻转 kernel)

    为什么是 depthwise?
      权重 shape (C_out, C_in, K) = (128, 1, 10):
      C_in=1 且 C_out=C_in*group → 每个输出通道只和自己那一个输入通道卷积,
      通道之间不互相累加 → 循环里不用再嵌套 C_in 那层 (普通conv要)。

    参数:
      x:  激活  [C, L]    本实验 = [128, 4]   (C=128通道, L=4时间步)
      w:  权重  [C, K]    本实验 = [128, 10]  (每通道10个tap的滤波器)
      pl: 左边补0个数, pr: 右边补0个数
    返回:
      y:  [C, L_out], L_out = L + pl + pr - K + 1

    原理: 滑动窗口点积
      每个输出位置 n, 从补零后的序列取一段长度 K 的窗口,
      与权重 w[c] 逐位相乘再求和 (点积)。
      窗口从 0 滑到 L_out-1, 共产生 L_out 个输出。
    """
    C, L = x.shape  # 解包: C=128, L=4
    K = w.shape[1]  # K=10, kernel 长度
    L_out = L + pl + pr - K + 1  # 卷积输出长度公式
    y = np.zeros((C, L_out), dtype=np.float64)  # float64 排除浮点累积误差

    for c in range(C):  # ── 外层: 每个通道独立 ──
        # 第c通道序列, 左边补pl个0, 右边补pr个0
        # 补零目的: 让kernel能覆盖序列边缘, 并控制输出长度
        # ump 的 tensor 只有 [128,4] 这 4 个真实值，padding 的 0 不是数据，dump 不会存
        xp = np.pad(x[c], (pl, pr))
        for n in range(L_out):  # ── 内层: 窗口滑动 ──
            # 取窗口 xp[n:n+K] (切片, 长度K) 与权重 w[c] 点积
            # 例: n=0→xp[0:K], n=1→xp[1:K+1], ... 窗口右移一步算一次
            y[c, n] = np.dot(w[c], xp[n : n + K])
    return y


def snr_db(ref, hat):
    """SNR(dB) = 20*log10(信号能量/误差能量), 越大越准"""
    err = ref - hat
    return 20 * np.log10(np.linalg.norm(ref) / np.linalg.norm(err))


if __name__ == "__main__":
    # ══════════ Step 0: 加载真实数据 ══════════
    # 数据来源: onnx_layer_dump 工具在 ORT 跑模型时逐层导出的中间值
    # 选这条链路: Slice_1 → conv_left (fsmn 记忆模块, 查 layers/layer_0033_Conv.json 确认)
    #   Slice_1 的输出 (13帧) = conv_left 的输入 x
    #   conv_left 的权重 = w
    #   conv_left 的输出 = y_ref (浮点真值)
    x, mx = load_tensor(RD / "_backbone_fsmn_0_fsmn_0_1_Slice_1_output_0.bin")
    w, mw = load_tensor(WD / "backbone_fsmn_0_1_conv_left_weight.bin")
    y_ref, my = load_tensor(
        RD / "_backbone_fsmn_0_fsmn_0_1_conv_left_Conv_output_0.bin"
    )
    # dump 是4维 [N,C,L,1], 去掉多余维 → [C,L]
    x = x[0, :, :, 0]  # [1,128,4,1] → [128,4]
    w = w[:, 0, :, 0]  # [128,1,10,1] → [128,10]
    y_ref = y_ref[0, :, :, 0]  # [1,128,4,1] → [128,4]
    C, L, K = x.shape[0], x.shape[1], w.shape[1]
    print(f"x: {x.shape} 范围[{x.min():.2f},{x.max():.2f}]  (真实激活)")
    print(f"w: {w.shape} 范围[{w.min():.3f},{w.max():.3f}]  (真实权重)")
    print(f"y_ref: {y_ref.shape} 范围[{y_ref.min():.2f},{y_ref.max():.2f}]  (真实输出)")
    # x: (128, 13) 范围（-14.38， 25.96）
    # w: (128, 10) 范围（-0.888， 0.522）
    # y：（128，4） 范围（-15.29，10.38）
    # y_h = (if_h + pad_top + pad_bottom - kernel_h) / stride_h + 1 ==> (13 + 0 - 10) / 1 + 1 = 4
    # y_w = (if_w + pad_left + pad_right - kernel_w) / stride_w + 1 ==> (1 + 0 - 1) / 1 + 1 = 1
    # ══════════ Step 1: 浮点复现 ══════════
    # 目标: numpy 手写 depthwise conv, 精确复现 dump 的 y_ref
    # 图结构来自 layers/layer_0033_Conv.json:
    #   conv_left 的 pads = [0,0,0,0] (无padding!), group=128, stride=1
    #   输入 Slice_1 是 13 帧 → L_out = 13 - 10 + 1 = 4 ✓ 和 dump 一致
    # 教训: 不能靠 tensor 名字猜链路 (Transpose 是喂 conv_right 的),
    #       必须查 layers/*.json 拿真实输入和 attrs
    pl, pr = 0, 0
    y_f = depthwise_conv1d(x, w, pl, pr)
    rmse = np.sqrt(np.mean((y_f - y_ref) ** 2))
    print(f"\n{'='*64}")
    print(
        f"Step 1: 浮点复现 → 输入 {L} 帧, kernel {K}, padding 左{pl}右{pr}, "
        f"输出 {y_f.shape[1]} 帧, RMSE = {rmse:.2e}"
    )
    print(f"  → 误差~1e-6 说明 depthwise conv 语义与 ONNX 完全一致")

    # ══════════ Step 2: 量化 ══════════
    # 激活: per-tensor 对称 (激活逐层不同, 一层一个scale就够)
    # 权重: per-channel 对称 (通道间尺度差14倍, 见实验3)
    scale_x = np.abs(x).max() / 127.0
    print(scale_x)
    x_q = np.clip(np.round(x / scale_x), -127, 127).astype(np.int16)
    # w.shape  = (128, 10)   # 128 个通道, 每通道 10 个权重值
    # scales_w = np.abs(w).max(axis=1) / 127.0    # → 形状 (128,)  ← 我们要的 沿通道内（axis=1，每行的 10 个权重）求最大 → 128 个值，每通道一个 scale
    # scales_w = np.abs(w).max() / 127.0          # → 形状 () 标量 ← per-tensor 版
    scales_w = (
        np.abs(w).max(axis=1) / 127.0
    )  # 深度卷积里，每个输出通道只用自己那个通道的权重。这个模型权重是 depthwise（w.shape[0]=128 通道，互不混合），所以 128 个通道的权重值域各不相同
    w_q = np.clip(np.round(w / scales_w[:, None]), -127, 127).astype(np.int16)
    print(f"\n{'='*64}")
    print(
        f"Step 2: 量化  scale_x={scale_x:.5f} (per-tensor), "
        f"scale_w ∈ [{scales_w.min():.5f},{scales_w.max():.5f}] (per-channel)"
    )

    # ══════════ Step 3: 整数卷积 (int32 累加) ══════════
    # 为什么必须 int32? 两个理由:
    #   a) 理论边界: 单个乘积 |w_q*x_q| ≤ 127*127=16129; 10个求和 → ±161290,
    #      远超 int16 上限 ±32767 → 硬件必须按最坏情况用 int32
    #   b) 实际数据: 下面打印真实范围 (可能不到边界, 但设计必须留余量)
    # 硬件里就是 NEON sadalp (int8向量乘加 + int32宽累加) — 呼应 Lesson 20
    L_out = L + pl + pr - K + 1
    y_int32 = np.zeros((C, L_out), dtype=np.int32)
    for c in range(C):
        xp = np.pad(x_q[c], (pl, pr))
        for n in range(L_out):
            # 先转 int32 再点积 → 乘积和累加全程 int32, 不溢出
            y_int32[c, n] = np.dot(
                w_q[c].astype(np.int32), xp[n : n + K].astype(np.int32)
            )
    theo_max = K * 127 * 127
    #   theo_max = K * 127 * 127    ->  theo_max = 161290证明必须用 int32
    #              │      │    └── 127: 权重 w_q 的绝对值上界
    #              │      └─────── 127: 激活 x_q 的绝对值上界
    #              └────────────── K:   卷积窗口里有几个乘积要加起来 (K=10)

    print(f"Step 3: 整数 conv → y_int32 ∈ [{y_int32.min()}, {y_int32.max()}]")
    print(f"  → 理论边界 ±{theo_max} (K*127*127), 实际峰值 ±{np.abs(y_int32).max()}")
    print(
        f"  → int16 上限 ±32767 → {'⚠ 必须 int32' if theo_max > 32767 else 'int16 够用'}"
    )

    # ---- 溢出演示: 故意用 int16 累加 (每次乘积都在 int16 里溢出) ----
    y_int16 = np.zeros((C, L_out), dtype=np.int16)
    for c in range(C):
        xp = np.pad(x_q[c], (pl, pr))
        for n in range(L_out):
            acc = np.int16(0)
            for k in range(K):
                acc = acc + np.int16(w_q[c, k]) * np.int16(xp[n + k])
            y_int16[c, n] = acc
    wrong = np.mean(y_int16 != y_int32.astype(np.int16))
    print(
        f"  溢出演示: int16 累加错位比例 = {wrong*100:.1f}%  ← 这就是为什么硬件要 int32 累加器"
    )

    # ══════════ Step 4: 反量化 (int32 → 浮点, 没除以输出scale) ══════════
    # ##################################################################################
    # 关键恒等式:
    #   浮点卷积 y_f = Σ(w_f·x_f) = Σ((w_q·scale_w)·(x_q·scale_x))
    #                = (scale_w·scale_x) · Σ(w_q·x_q) = (scale_w·scale_x)·y_int32
    # ##################################################################################
    # 所以 y_int32 的"浮点还原scale" = scale_w·scale_x (每通道)
    #   A: 直接浮点乘 → 数学恒等, 误差=纯量化误差 (理论最好成绩)
    #   B: 定标逼近 → 浮点除法 mult/2^shift 复现 scale, 验证16bit逼近无损
    #   B2: ⚠ 错误示范 → 在浮点小数值域用 >> 取整! 取整误差~0.5 相对输出
    #       幅度(~10)太大 → SNR 掉到 17dB。硬件正确做法是 D: 先除以 scale_y
    #       放大到 int8 域(±127)再取整, 0.5 相对 127 就微不足道
    scale_yA = scales_w[:, None] * scale_x
    y_A = y_int32.astype(np.float64) * scale_yA

    mults = np.zeros(C, dtype=np.int64)
    shifts = np.zeros(C, dtype=np.int64)
    for i in range(C):
        mults[i], shifts[i] = scale_to_fixed(scales_w[i] * scale_x, 16)
    # B: 定标逼近 —— 浮点除法复现 mult/2^shift (验证16bit定标不损失精度)
    y_B = y_int32.astype(np.float64) * (mults / (1 << shifts)).reshape(-1, 1)
    # B2: 错误示范 —— 浮点小数值域 >> 取整 (取整误差~0.5, 相对幅度~10 太大)
    y_B2 = (
        (
            y_int32.astype(np.int64) * mults[:, None]
            + (np.int64(1) << (shifts[:, None] - 1))
        )
        >> shifts[:, None]
    ).astype(np.float64)

    print(f"\n{'='*64}")
    print(f"Step 4: 反量化到浮点 → y_A (理论) / y_B (定标)")

    # ══════════ Step 5: 完整 requantize → int8 输出 (关键!) ══════════
    # 完整推理的输出也得是 int8 (给下一层当输入), 所以还要除以输出 scale:
    #   浮点输出  y_f  = y_int32 · scale_w · scale_x
    #   输出量化  y_q  = round(y_f / scale_y)          ← 除以输出 scale
    #   合并一步   y_q = round(y_int32 · (scale_w·scale_x / scale_y))
    #                              └────────┬─────────┘
    #                          requantize 合并 multiplier (定标成一个mult+shift)
    # scale_w·scale_x 负责"还原回浮点", scale_y 负责"再量化成int8",
    # 硬件把两个动作合成一个整数乘加移位 — 就是 QNN 每个算子 requantize 参数来源
    #
    # scale_y 从哪来? → PTQ 校准: 拿真实数据统计输出的 min/max
    # 这里直接用 dump 真实输出统计 (per-channel, 和权重一致)
    scale_y = np.abs(y_ref).max(axis=1, keepdims=True) / 127.0
    y_q_ref = np.clip(np.round(y_ref / scale_y), -127, 127).astype(np.int16)

    # D: 完整 requantize (合并 multiplier, 定标实现)
    mults_rq = np.zeros(C, dtype=np.int64)
    shifts_rq = np.zeros(C, dtype=np.int64)
    for i in range(C):
        # 合并 multiplier = scale_w·scale_x / scale_y (比1小 → 需要右移, shift>0)
        s = scales_w[i] * scale_x / scale_y[i, 0]
        mults_rq[i], shifts_rq[i] = scale_to_fixed(s, 16)
        if shifts_rq[i] == 0:
            shifts_rq[i] = 1  # 保险: 本数据不会出现
    y_q = (
        (
            y_int32.astype(np.int64) * mults_rq[:, None]
            + (np.int64(1) << (shifts_rq[:, None] - 1))
        )
        >> shifts_rq[:, None]
    ).astype(np.int16)
    y_q = np.clip(y_q, -127, 127)  # int8 输出也要 clamp

    print(f"\n{'='*64}")
    print(f"Step 5: 完整 requantize → int8 输出 (除以输出 scale_y)")
    print(f"  scale_y ∈ [{scale_y.min():.5f},{scale_y.max():.5f}] (校准: 输出min/max)")
    print(
        f"  requant mult ∈ [{mults_rq.min()},{mults_rq.max()}], "
        f"shift ∈ [{shifts_rq.min()},{shifts_rq.max()}]"
    )
    match = np.mean(y_q == y_q_ref)
    print(f"  int8 输出一致率 = {match*100:.2f}%  (y_q vs round(y_ref/scale_y))")
    print(f"  注: 一致率不到100%正常 — 量化误差会让边界附近的点翻转, 看SNR更本质")
    print(f"  int8 域 SNR = {snr_db(y_q_ref, y_q):.2f} dB")

    # ══════════ Step 6: 汇总 ══════════
    # y_q 是 int8 域, 反量化回浮点再和真值比 (乘回 scale_y)
    y_q_float = y_q.astype(np.float64) * scale_y
    print(f"\n{'='*64}")
    print("汇总: 各环节精度 (对比浮点真值 y_ref)")
    print(f"  {'方法':<32} {'SNR(dB)':>10} {'最大绝对误差':>14}")
    for tag, y_hat in [
        ("A 反量化 scale传播(浮点)", y_A),
        ("B 定标逼近 mult/2^shift(浮点除)", y_B),
        ("B2 错误示范: 浮点域>>取整", y_B2),
        ("D 完整requantize→int8(还原)", y_q_float),
    ]:
        print(
            f"  {tag:<32} {snr_db(y_ref, y_hat):>10.2f} {np.abs(y_ref-y_hat).max():>14.3e}"
        )

    print(f"\n{'='*64}")
    print("结论:")
    print("  1. A ≈ B → 16bit 定标逼近无损 (实验3结论在真实数据复现)")
    print("  2. D ≈ A → 完整 requantize(除以输出scale) 也只损失量化误差")
    print(
        "  3. B2 << 其它 → 取整必须在 int8 域做 (先除scale_y放大), "
        "不能在浮点小数值域>>取整"
    )
    print("  4. 全程整数: 输入int8 → 权重int8 → 累加int32 → requant(mult+shift)")
    print("     → 输出int8。没有任何浮点, 这就是部署到 NPU/DSP 的样子")
