# gen_model.py
# Lesson 11: 性能分析演示模型 —— 3 层 MLP 分类器
# 结构: X[1,256] → Gemm(256→1024)+Relu → Gemm(1024→256)+Relu → Gemm(256→10) → Softmax → Y[1,10]
# 故意用中等规模权重，让 MatMul/Gemm 成为热点，profiling 有区分度

import onnx
from onnx import helper, TensorProto
import numpy as np


def make_gemm(name, x, w, b, y):
    return helper.make_node(
        "Gemm", inputs=[x, w, b], outputs=[y], name=name, alpha=1.0, beta=1.0, transB=1
    )


rng = np.random.default_rng(42)

# ---- 权重 (transB=1 所以 W 是 [out, in]) ----
w1 = rng.standard_normal((1024, 256), dtype=np.float32) * 0.1
b1 = rng.standard_normal((1024,), dtype=np.float32) * 0.1
w2 = rng.standard_normal((256, 1024), dtype=np.float32) * 0.1
b2 = rng.standard_normal((256,), dtype=np.float32) * 0.1
w3 = rng.standard_normal((10, 256), dtype=np.float32) * 0.1
b3 = rng.standard_normal((10,), dtype=np.float32) * 0.1

inits = []
for name, arr in [
    ("W1", w1),
    ("B1", b1),
    ("W2", w2),
    ("B2", b2),
    ("W3", w3),
    ("B3", b3),
]:
    inits.append(
        helper.make_tensor(
            name, TensorProto.FLOAT, list(arr.shape), arr.flatten().tolist()
        )
    )

# ---- 节点 ----
nodes = [
    make_gemm("Gemm1", "X", "W1", "B1", "h1"),  # [1,1024]
    helper.make_node("Relu", ["h1"], ["a1"], name="Relu1"),
    make_gemm("Gemm2", "a1", "W2", "B2", "h2"),  # [1,256]
    helper.make_node("Relu", ["h2"], ["a2"], name="Relu2"),
    make_gemm("Gemm3", "a2", "W3", "B3", "logits"),  # [1,10]
    helper.make_node("Softmax", ["logits"], ["Y"], name="Softmax1", axis=1),
]

# ---- graph ----
graph = helper.make_graph(
    nodes,
    "mlp3_graph",
    inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 256])],
    outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])],
    initializer=inits,
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx.save(model, "mlp3.onnx")
print("Saved: mlp3.onnx (6 nodes: 3xGemm, 2xRelu, Softmax)")
