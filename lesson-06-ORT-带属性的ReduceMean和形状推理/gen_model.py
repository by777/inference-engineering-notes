# gen_model.py
# Lesson 06: 生成带属性的 ReduceMean 和形状推理模型

# 用法：
# python3 gen_model.py axis0_keep0
# python3 gen_model.py axis1_keep0
# python3 gen_model.py axis1_keep1

import sys
import onnx
from onnx import helper, TensorProto

case_name = sys.argv[1] if len(sys.argv) > 1 else "axis1_keep0"

case_map = {
    "axis0_keep0": {"axis": 0, "keepdims": 0},
    "axis1_keep0": {"axis": 1, "keepdims": 0},
    "axis1_keep1": {"axis": 1, "keepdims": 1},
}

if case_name not in case_map:
    raise ValueError(f"Unknown case name: {case_name}")

axis = case_map[case_name]["axis"]
keepdims = case_map[case_name]["keepdims"]

# node -> graph -> model
node = helper.make_node(
    "MyReduceMean",
    inputs=["X"],
    outputs=["Y"],
    domain="my_domain",
    axis=axis,  # 这两个参数可以自己随便编
    keepdims=keepdims,
)

# 形状推理，
input_shape = [2, 3]
if axis == 0 and keepdims == 0:
    output_shape = [3]
elif axis == 1 and keepdims == 0:
    output_shape = [2]
elif axis == 1 and keepdims == 1:
    output_shape = [2, 1]
else:
    raise ValueError(f"Unsupported axis={axis}, keepdims={keepdims}")

# graph
graph = helper.make_graph(
    [node],
    "my_reduce_mean_graph",
    [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape),
    ],
    [
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape),
    ],
)


# model
model = helper.make_model(
    graph,
    opset_imports=[
        helper.make_opsetid("", 17),
        helper.make_opsetid("my_domain", 1),
    ],
)
model_file = f"my_reduce_mean_{case_name}.onnx"
onnx.save(model, model_file)
print(f"The model is saved to {model_file}")
