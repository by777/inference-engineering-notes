# gen_model.py
# Lesson 08: C API 推理
# 生成 Y = X * 2 + 1 的简单模型（纯标准算子，无需自定义算子）
import onnx
from onnx import helper, TensorProto
import numpy as np

# 构造节点: Mul(X, init_2) → MulOut, 然后 Add(MulOut, init_1) → Y
init_2 = helper.make_tensor("init_2", TensorProto.FLOAT, [1], [2.0])
init_1 = helper.make_tensor("init_1", TensorProto.FLOAT, [1], [1.0])


node_mul = helper.make_node(
    "Mul",
    inputs=["X", "init_2"],
    outputs=["MulOut"],
)

node_add = helper.make_node(
    "Add",
    inputs=["MulOut", "init_1"],
    outputs=["Y"],
)

#  graph
graph = helper.make_graph(
    nodes=[node_mul, node_add],
    name="SimpleGraph",
    inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
    outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
    initializer=[init_2, init_1],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx.save(model, "my_test_model.onnx")
print("Saved: my_test_model.onnx")
