# gen_model.py
# 生成一个简单的测试模型：Y = X * 2 + 1
# 用于 SessionOptions 和图优化对比实验

import onnx
from onnx import helper, TensorProto

# 构造节点
mul_node = helper.make_node(
    "Mul",
    inputs=["X", "two"],  # ← 从"名字叫 X" 和 "名字叫 two" 的 tensor 读数据
    outputs=["scaled"],  # ← 计算结果写入"名字叫 scaled" 的 tensor
    domain="",
)
add_node = helper.make_node(
    "Add",
    inputs=["scaled", "one"],
    outputs=["Y"],
    domain="",
)

# 常量
#
#  名字          float 类型      1维     值 = 2.0
two_init = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
one_init = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])

# 构造计算图
# ONNX 图不是靠"指针连接"来描述数据流的，而是靠名字匹配
#           ┌── two (常量=2.0) ──┐
# X ────────┤                     ├── scaled ──┬── one (常量=1.0) ──┐
#           └──── Mul 节点 ───────┘            │                     ├── Y
#                                              └──── Add 节点 ───────┘
graph = helper.make_graph(
    [mul_node, add_node],
    "y_eq_x_mul_2_add_1",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],  # 输入张量占位符
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],  # 输出张量占位符
    initializer=[two_init, one_init],  # 是图的常量表
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx.save(model, "my_test_model.onnx")
print("saved my_test_model.onnx")
