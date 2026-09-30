# gen_model.py
# Lesson 03: 生成带属性的自定义算子模型
# MyAdd(X, Y) -> Z, domain="my_domain", attribute scale=2.5

import onnx
from onnx import helper, TensorProto

# 构造节点，加上属性scale=2.5
node = helper.make_node(
    "MyAdd",  # 算子类型
    inputs=["X", "Y"],  # 输入
    outputs=["Z"],  # 输出
    domain="my_domain",  # 自定义算子域
    scale=2.5,  # 属性
)

# 构造计算图
graph = helper.make_graph(
    [node],  # 节点列表
    "my_add_graph",  # 图名称
    [  # 输入张量
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4]),
    ],
    [  # 输出张量
        helper.make_tensor_value_info("Z", TensorProto.FLOAT, [4]),
    ],
)
#构造模型
model = helper.make_model(
    graph,
    opset_imports=[ #  是 ONNX 模型的算子集版本声明表
        helper.make_opsetid("", 17),  # 标准 ONNX domain, opset 17
        helper.make_opsetid("my_domain", 1),  # 自定义 domain, opset 1
    ],
)
onnx.save(model, "my_add.onnx")
print("The model is saved to my_add.onnx")