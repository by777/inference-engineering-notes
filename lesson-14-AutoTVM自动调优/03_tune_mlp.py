# 03_tune_mlp.py
# Lesson 14: AutoTVM 自动调优 - 实验 3: 对 mlp3.onnx 整图自动调优
#
# 实验 2 是手写模板调 GEMM；这次是"真模型"——从 ONNX 图自动提取任务
# 流程:
#   ONNX → Relay → 提取可调优的算子任务 → 逐个调优 → 用最优配置编译 → 对比 ORT
#
# 运行: python 03_tune_mlp.py  (约 3-5 分钟)
# 前置: 目录下需要有 mlp3.onnx

# ========== 环境配置（必须在 import tvm 之前）==========
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tvm-src" / "python"))
sys.path.insert(0, str(_REPO / "tvm-bin"))
os.environ["LD_LIBRARY_PATH"] = (
    str(_REPO / "tvm-bin") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ["TVM_NUM_THREADS"] = "16"

import time
import numpy as np
import tvm
from tvm import relay, autotvm
from tvm.contrib import graph_executor
import onnx

# --------- 1. ONNX->Relay --------- #
onnx_model = onnx.load("mlp3.onnx")
mod, params = relay.frontend.from_onnx(onnx_model)
target = "llvm"

# --------- 2. 提取可调优任务 --------- #
# 从Relay图中自动找出可调优算子如Dense，Softmax
# 这一步不需要写模板，TVM内置了常见算子的搜索模板
# TOPI（TVM Operator Inventory）已经覆盖了绝大多数常见算子：
# 类别	算子
# 矩阵类	dense、matmul、batch_matmul、conv2d、conv1d、conv3d
# 池化/激活	pool、relu、sigmoid、softmax
# 逐元素	add、mul、exp、tanh、clip
# 归一化	batch_norm、layer_norm
# 其他	transpose、reshape、concat、split、reduce
# 这些算子的调度模板、搜索空间TVM 都写好了。你只要：
# tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
# 以下情况 TVM 没有现成模板，必须手写：
# 自定义算子、融合了奇怪结构的注意力、新型卷积变体、
# 几个算子组合成一个 kernel（比如 Norm+量化）、NPU 上才有的算子，TVM 上游没实现
# 此时你要自己定义搜索空间：
# @autotvm.template("my_fancy_op")
# def my_op_template():
#     # 1. 定义计算
#     C = te.compute(..., name="C")
#     # 2. 声明哪些参数可搜索
#     cfg.define_knob("tile", [1, 2, 4, 8, 16])
#     # 3. 用 cfg 的值做调度
#     ...
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
print(f"提取到 {len(tasks)} 个可调优任务:")
for t in tasks:
    print(f"  - {t.name}  搜索空间: {len(t.config_space)} 种")

# -------- 3. 调优每个任务 --------- #
log_file = "mlp3_autotvm.log"
tuner = autotvm.tuner.XGBTuner(tasks[0])  # 选择调优器
print(f"调优任务: {tasks[0].name}  搜索空间: {len(tasks[0].config_space)} 种")
tuner.tune(
    n_trial=200,
    measure_option=autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=5,  # 每轮跑 5 次取平均
            repeat=1,  # 整个测量过程只做 1 轮
            min_repeat_ms=100,  # 如果 number=5 只花了很少时间（比如 GEMM 太快，5 次才 1ms），
            # 那计时精度不够——自动加大 number，直到总耗时 ≥ 100ms
        ),
    ),
    callbacks=[autotvm.callback.log_to_file(log_file)],
)
print("调优完成")

# -------- 4. 用最优配置编译 --------- #
with autotvm.apply_history_best(log_file):
    with tvm.target.Target(target):
        lib = relay.build(mod, target=target, params=params)

# -------- 5. 执行，对比ORT --------- #
dev = tvm.cpu()
m = graph_executor.GraphModule(lib["default"](dev))
rng = np.random.default_rng(42)
x = rng.standard_normal((1, 256), dtype=np.float32) * 0.5
m.set_input("X", tvm.nd.array(x))
# 计时（100 次平均）
m.run()  # 预热
t0 = time.time()
for _ in range(100):
    m.run()
tvm_ms = (time.time() - t0) / 100 * 1000

tvm_out = m.get_output(0).numpy()

# ORT对比
import onnxruntime as ort

ort_sess = ort.InferenceSession("mlp3.onnx")
ort_out = ort_sess.run(None, {"X": x})[0]
diff = np.max(np.abs(tvm_out - ort_out))
print(f"\nAutoTVM 整图推理: {tvm_ms:.4f} ms")
print(f"TVM vs ORT 误差: {diff:.2e}  ({'✅' if diff < 1e-4 else '❌'})")
print(f"\n参考: Lesson 13 未调优 TVM vs ORT 也是 ~0.2ms 级别，调优后应更快")
