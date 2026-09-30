# gen_model.py
# Lesson 05: 生成多输入多输出自定义算子模型
# MyAddSub(X, Y) -> (Sum, Diff)

import onnx
from onnx import helper, TensorProto

# node -> graph -> model
# 给“这个节点本身”贴标签。
# ORT 才知道它不是标准 ONNX 算子，而是你自己注册的自定义算子
node = helper.make_node(
    "MyAddSub",
    inputs=["X", "Y"],
    outputs=["Sum", "Diff"],
    domain="my_domain",
)
graph = helper.make_graph(
    [node],
    "my_addsub_graph",
    [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4]),
    ],
    [
        helper.make_tensor_value_info("Sum", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("Diff", TensorProto.FLOAT, [4]),
    ],
)


model = helper.make_model(
    graph,
    opset_imports=[
        helper.make_opsetid("", 17),
        helper.make_opsetid(
            "my_domain", 1
        ),  # 在声明：这个模型用到了哪些 domain，以及各自用的版本号是多少
    ],
)
onnx.save(model, "my_add_sub.onnx")
print("The model is saved to my_add_sub.onnx")
