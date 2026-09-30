# A0_gen_model.py
# Lesson 22 实验A: 生成一个含 depthwise conv 的最小 onnx 模型
# 用于跟踪 MNN 的 CPUConvolutionDepthwise 调用链
# depthwise conv: 输入 [1,3,8,8] → 权重 [3,1,3,3] → 输出 [1,3,6,6]
# 注意: ONNX 里 depthwise = group=C_in 的 Conv

import onnx
from onnx import helper, TensorProto
import numpy as np

# ---- 输入 N C H W----
x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])

# ---- 权重 OC IC KH KW(depthwise: group=3, C_in=3, C_out=3, 每通道一个 3x3 kernel) ----
w_data = np.random.RandomState(0).randn(3, 1, 3, 3).astype(np.float32)
w_init = helper.make_tensor(
    "weight", TensorProto.FLOAT, [3, 1, 3, 3], w_data.flatten().tolist()
)

# ---- Conv 节点 (group=3 → depthwise) ----
conv = helper.make_node(
    "Conv",
    inputs=["input", "weight"],
    outputs=["output"],
    kernel_shape=[3, 3],
    pads=[1, 1, 1, 1],
    strides=[1, 1],
    group=3,
)

# ---- 输出 ----
y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])

graph = helper.make_graph([conv], "dwconv", [x], [y], [w_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8

onnx.checker.check_model(model)
onnx.save(model, "dwconv.onnx")
print("生成 dwconv.onnx: depthwise conv, input [1,3,8,8], group=3, kernel 3x3, pads=1")
