# 02_from_onnx.py
# Lesson 13: AI 编译器入门（TVM） - 实验2：整图编译 ONNX → Relay → 执行
#
# 对应编译器架构：前端（把 ONNX 翻译成 Relay IR）
# 之前实验1是"手写算子"（算子级编译）
# 这次是"整图编译"（从 ONNX 模型文件到可执行模块）
#
# 运行： python 02_from_onnx.py

import numpy as np
import onnx
import tvm
from tvm import relay
from tvm.contrib import graph_executor

# ========== 1. 前端：ONNX -> Relay IR ==========
# relay.frontend.from_onnx() 是TVM的ONNX前端
# 作用：读懂ONNX图的每个节点，翻译成Relay的表达式（函数式IR）
# 输出mod（IRModule）+ params（权重字典）
print("===== 1. ONNX → Relay =====")
onnx_model = onnx.load("mlp3.onnx")
mod, params = relay.frontend.from_onnx(onnx_model)

print("mod = ", mod)  # 打印 Relay IR：能看到 Gemm 被翻译成 nn.dense + add
print("params = ", params)  # 为空
print()


# ========= 2. 中端 + 后端：relay.build(pass优化 + 代码生成) ==========
# opt_level = 3: 最高级别优化：包含常量折叠，算子融合等
# target = "llvm"：生成x86-64 CPU代码
print("===== 2. relay.build 编译 =====")
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
print("编译完成:", lib)
print()

# ============ 3. 创建图执行器 ============
# graph_executor.GraphModule: TVM的图执行器（类似 ORT 的 InferenceSession）
# lib["default"]：默认的编译产物模块
print("===== 3. 创建图执行器 =====")
dev = tvm.cpu()
m = graph_executor.GraphModule(lib["default"](dev))

# ============ 4. 输入并执行 ============
rng = np.random.default_rng(seed=42)
x = rng.standard_normal((1, 256), dtype=np.float32) * 0.5
m.set_input("X", tvm.nd.array(x))  # 按名字设置输入（和 ORT 的 feed dict 类似）
m.run()
tvm_out = m.get_output(0).numpy()
print("TVM 输出 (前5个):", tvm_out[0][:5])


# ============ 5. 对比 ONNX Runtime ============
import onnxruntime as ort

sess = ort.InferenceSession("mlp3.onnx", providers=["CPUExecutionProvider"])
ort_out = sess.run(["Y"], {"X": x})[0]

diff = np.max(np.abs(tvm_out - ort_out))
print(f"\nTVM vs ORT 最大绝对误差: {diff:.2e}")
print("====== 一致 ======" if diff < 1e-4 else "====== 不一致 ======")
