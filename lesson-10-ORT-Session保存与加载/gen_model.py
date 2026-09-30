# gen_model.py
# Lesson 10: Session 保存/加载演示模型
# 图结构: MatMul → Add → Relu（经典的可融合模式）
#   ORT 的图优化会把 MatMul+Add+Relu 融合成 1 个 FusedGemm 节点
#   → 优化前后节点数 3 → 1，方便看出"优化后模型"的差别

import onnx
from onnx import helper, TensorProto
import numpy as np

# ---- initializers ----
w = np.arange(32, dtype=np.float32).reshape(8, 4)
init_w = helper.make_tensor("W", TensorProto.FLOAT, [8, 4], w.flatten().tolist())
b = np.arange(4, dtype=np.float32)  # [0,1,2,3]
init_b = helper.make_tensor("B", TensorProto.FLOAT, [4], b.flatten().tolist())

# ---- nodes: MatMul -> Add -> Relu ----
node_mm = helper.make_node("MatMul", inputs=["X", "W"], outputs=["mm_out"])
node_add = helper.make_node("Add", inputs=["mm_out", "B"], outputs=["add_out"])
node_relu = helper.make_node("Relu", inputs=["add_out"], outputs=["Y"])

# ---- graph ----
graph = helper.make_graph(
    [node_mm, node_add, node_relu],
    "save_load_graph",
    inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])],
    outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
    initializer=[init_w, init_b],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx.save(model, "original.onnx")
print("Saved: original.onnx (3 nodes: MatMul, Add, Relu)")
