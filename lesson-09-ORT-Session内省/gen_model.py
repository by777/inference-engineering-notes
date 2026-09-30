# gen_model.py
# Lesson 09: Session 内省演示模型
# 图结构:
#   X [1, N] (N 是 symbolic 维度)   W [8,4] initializer   B [4] 既是 input 又是 initializer
#   Gemm(X, W, B, alpha=1.0) -> gemm_out
#   Relu(gemm_out) -> y1
# 故意包含: symbolic 维度、可覆盖 initializer、节点属性、model metadata

import onnx
from onnx import helper, TensorProto
import numpy as np

# ---- initializers ----
# W: [8, 4] 固定的权重
w = np.arange(32, dtype=np.float32).reshape(8, 4)
init_w = helper.make_tensor("W", TensorProto.FLOAT, [8, 4], w.flatten().tolist())
# B: [4] 既是 initializer，又声明为 graph input => "overridable initializer"
b = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
init_b = helper.make_tensor("B", TensorProto.FLOAT, [4], b.flatten().tolist())

# ---- nodes ----
# Gemm: Y = alpha * X @ W + beta * B，alpha/beta 是节点属性
node_gemm = helper.make_node(
    "Gemm",
    inputs=["X", "W", "B"],
    outputs=["gemm_out"],
    alpha=1.0,  # 属性
    beta=1.0,  # 属性
    transB=0,  # 属性
)
node_relu = helper.make_node("Relu", inputs=["gemm_out"], outputs=["y1"])

# ---- graph ----
# X 的第二维用 symbolic 名 "N"，展示 C API 如何获取动态维度名
graph = helper.make_graph(
    [node_gemm, node_relu],
    "introspect_graph",
    inputs=[
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, "N"]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [4]),
    ],
    outputs=[
        helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 4]),
    ],
    initializer=[init_w, init_b],
)

# ---- model + metadata ----
model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 17)],
    producer_name="lesson-09",
    producer_version="1.0.0",
)
model.model_version = 3
model.graph.doc_string = "Session introspection demo"
model.description = "lesson 09 demo"
helper.set_model_props(model, {"model_type": "demo", "author": "me"})

onnx.save(model, "my_introspect_model.onnx")
print("Saved: my_introspect_model.onnx")
