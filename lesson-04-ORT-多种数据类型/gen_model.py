# gen_model.py
# Lesson 04: 多种数据类型的自定义算子
# MyAdd 支持 float32、float64、int32、int64

import onnx
from onnx import helper, TensorProto
import sys

# 命令行参数：python gen_model.py [float|double|int32|int64]
# 默认 float
dtype_arg = sys.argv[1] if len(sys.argv) > 1 else "float"

# 映射关系
dtype_map = {
    "float": TensorProto.FLOAT,
    "double": TensorProto.DOUBLE,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
}

if dtype_arg not in dtype_map:
    print(f"Unsupported dtype: {dtype_arg}")
    sys.exit(1)

proto_dtype = dtype_map[dtype_arg]
op_name_map = {
    "float": "MyAddFloat",
    "double": "MyAddDouble",
    "int32": "MyAddInt32",
    "int64": "MyAddInt64",
}
op_name = op_name_map[dtype_arg]

# 构造节点
node = helper.make_node(
    op_name,
    inputs=["X", "Y"],
    outputs=["Z"],
    domain="my_domain",
    scale=2.5,
)

# 构造计算图
graph = helper.make_graph(
    [node],
    "my_add_graph",
    [
        helper.make_tensor_value_info("X", proto_dtype, [4]),
        helper.make_tensor_value_info("Y", proto_dtype, [4]),
    ],
    [
        helper.make_tensor_value_info("Z", proto_dtype, [4]),
    ],
)

# 构造模型
model = helper.make_model(
    graph,
    opset_imports=[
        helper.make_opsetid("", 17),
        helper.make_opsetid("my_domain", 1),
    ],
)

model_file = f"my_add_{dtype_arg}.onnx"
onnx.save(model, model_file)
print(f"The model is saved to {model_file}")
