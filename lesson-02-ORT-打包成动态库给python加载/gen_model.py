# gen_model.py
# Lesson 01: 生成一个只含自定义算子 MyAdd 的 ONNX 模型
#
# 依赖: pip install onnx>=1.14 numpy
#       (onnx 会自动拉 numpy 做依赖)
import onnx
from onnx import helper, TensorProto

# ====== 1. 构造节点 ======
# make_node 参数: op_type, inputs, outputs, domain, ...
# 关键点: domain="my_domain" 告诉 ORT 这个算子不是标准 ONNX 算子,
#         要去 SessionOptions 上注册的自定义域里找 "MyAdd" 这个算子
node = helper.make_node("MyAdd", ["X", "Y"], ["Z"], domain="my_domain")
# ====== 2. 构造计算图 ======
# make_graph 参数: nodes, name, inputs, outputs
# make_tensor_value_info 参数: name, dtype, shape
#   X: 输入, float32, shape=[4]
#   Y: 输入, float32, shape=[4]
#   Z: 输出, float32, shape=[4]
graph = helper.make_graph(
    [node],  # 节点列表
    "my_add_graph",  # 图名
    [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4]),
    ],
    [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [4])],
)

# ====== 3. 构造模型 ======
# make_model 参数: graph, opset_imports
# opset_imports 必须列出模型用到的所有 domain:
#   ("", 17)         — 标准 ONNX domain(空字符串), opset 版本 17
#   ("my_domain", 1) — 我们自定义的 domain, 版本 1
#                      必须和 step1_myadd.cpp 里 CustomOpDomain{"my_domain"} 一致!
model = helper.make_model(
    graph,
    opset_imports=[
        helper.make_opsetid("", 17),  #  声明: 用到标准 ONNX 算子, opset 17
        helper.make_opsetid(
            "my_domain", 1
        ),  ## 声明: 用到自定义域 my_domain 算子, opset 1
    ],
)
onnx.save(model, "my_add.onnx")
print("The model is saved to my_add.onnx")
