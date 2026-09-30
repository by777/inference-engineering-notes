# 03_relay_pass.py
# Lesson 13: AI 编译器入门（TVM） - 实验3：Relay Pass 内省
#
# 对应编译器架构：中端（pass 优化）
# 目标：看 opt_level=3 到底对图做了什么 —— 特别是"算子融合"
#
# 概念：
#   融合（Fusion）：把多个相邻算子合成一个 kernel。
#   例如 dense + bias_add + relu 融合成一次循环，减少中间张量的内存读写。
#   TVM 里的 FuseOps pass 就干这个。
#
# 运行： python 03_relay_pass.py

import numpy as np
import onnx

import tvm
from tvm import relay
from tvm import transform  # ← Sequential 在这里
from tvm.contrib import graph_executor

# ============ 1. 载入模型 → Relay IR ============
onnx_model = onnx.load("mlp3.onnx")
mod, params = relay.frontend.from_onnx(onnx_model)

print("===== 1. 原始 Relay IR（融合前）=====")
print(mod)
print()

# ============ 2. 只跑 FuseOps pass，看融合效果 ============
# transform.FuseOps 是 TVM 的算子融合 pass
# 它会自动识别"可融合的算子组"（如 dense+add+relu），合成一个融合算子
print("===== 2. 应用 FuseOps pass =====")
seq = transform.Sequential(
    [
        relay.transform.FuseOps(),  # ← FuseOps 仍在 relay.transform
    ]
)
fused_mod = seq(mod)

print(fused_mod)
print()
# ============ 3. 对比：融合前后编译的性能 ============
# 用 graph_executor 分别跑融合前后，对比耗时
# （小模型差异可能不大，但能看出融合减少了节点数）


def build_and_run(mod, params, tag):
    with tvm.transform.PassContext(opt_level=0):  # 关掉自动优化，只保留我们的手动 pass
        lib = relay.build(mod, target="llvm", params=params)
    dev = tvm.cpu()
    m = graph_executor.GraphModule(lib["default"](dev))

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 256), dtype=np.float32) * 0.5
    m.set_input("X", tvm.nd.array(x))
    m.run()

    # 计时
    import time

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        m.run()
        times.append(time.perf_counter() - t0)
    avg_ms = np.mean(times) * 1000
    print(f"{tag}: 平均 {avg_ms:.4f} ms/iter")
    return avg_ms


t_fused = build_and_run(fused_mod, params, "融合后")

print("\n✅ 完成：对比两个 mod 的节点数量，可见融合减少了中间节点")
