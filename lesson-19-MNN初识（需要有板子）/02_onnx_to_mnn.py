# 02_onnx_to_mnn.py
# Lesson 19: MNN 初识 - 实验 2: onnx → .mnn 转换 + Python 推理
#
# 对比 TVM (Lesson 13):
#   TVM: relay.frontend.from_onnx(onnx_model) → IRModule → relay.build → 机器码
#   MNN: mnnconvert 转成 .mnn → Interpreter 加载执行（解释式）
#
# 关键区别（后面实验会反复用）:
#   TVM = 编译执行: 模型变成机器码 .so，运行时直接执行机器码
#   MNN = 解释执行: .mnn 是数据文件（图结构+权重），运行时逐算子解释执行
#
# 运行: source .venv/bin/activate && python 02_onnx_to_mnn.py

import os
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort

# ========== 1. ONNX → MNN 转换 ==========
onnx_path = "../lesson-13-AI编译器入门（TVM）/mlp3.onnx"
mnn_path = "mlp3.mnn"

print("=== 1. ONNX → MNN 转换 ===")
# mnnconvert 是 MNN 转换器命令行（setup.py 装进 venv/bin 的）
# 类似 TVM 的 relay.frontend.from_onnx（前端）
ret = subprocess.run(
    ["mnnconvert", "-f", "ONNX", "--modelFile", onnx_path, "--MNNModel", mnn_path],
    capture_output=True,
    text=True,
)
print(ret.stdout[-500:] if ret.stdout.strip() else "转换完成")
print("生成:", mnn_path, "大小:", os.path.getsize(mnn_path), "bytes")

# ========== 2. MNN Python 推理 ==========
print("\n=== 2. MNN Python 推理 ===")
import MNN

# Interpreter: 加载 .mnn（类比 ORT 的 InferenceSession / TVM 的 graph_executor）
interpreter = MNN.Interpreter(mnn_path)
# createSession: 创建推理会话（运行时做内存分配、图优化）
session = interpreter.createSession()
# 拿到输入/输出张量
input_tensor = interpreter.getSessionInput(session)
output_tensor = interpreter.getSessionOutput(session)
print("输入形状:", input_tensor.getShape(), "dtype:", input_tensor.getDataType())
print("输出形状:", output_tensor.getShape())

# 准备输入数据（固定 seed，和 ORT 用完全相同的数据）
np.random.seed(42)
x = np.random.rand(1, 256).astype("float32")

# MNN 输入要经过 copyFromHostTensor（内部布局是 NC4HW4，为了 NEON 优化）
# Tensor_DimensionType_Caffe = NCHW 布局（和 ONNX 一致）
# NN.Tensor_DimensionType_Caffe "Caffe"，但它不代表 Caffe 框架，而是代表一种维度顺序约定
# NCHW: [Batch, Channel, Height, Width]
#       样本数  通道数   高度    宽度
tmp_input = MNN.Tensor(
    (1, 256),  # 形状: [batch=1, features=256]
    MNN.Halide_Type_Float,  # 数据类型: float32
    x,  # 数据本体: numpy 数组（真正的内容）
    MNN.Tensor_DimensionType_Caffe,  # 数据布局: NCHW
)
input_tensor.copyFromHostTensor(tmp_input)

# 跑推理
interpreter.runSession(session)

# 取输出（copyToHostTensor: MNN 内部布局 → numpy）
output = MNN.Tensor(
    output_tensor.getShape(),
    MNN.Halide_Type_Float,
    np.zeros(output_tensor.getShape(), dtype="float32"),
    MNN.Tensor_DimensionType_Caffe,
)
output_tensor.copyToHostTensor(output)
mnn_out = np.array(output.getData()).reshape(output_tensor.getShape())

# ========== 3. ORT 推理（对比基准）==========
print("\n=== 3. ORT 推理（对比）===")
ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ort_out = ort_sess.run(None, {"X": x})[0]
print("ORT 输出前5:", np.round(ort_out[0][:5], 4))
print("MNN 输出前5:", np.round(mnn_out[0][:5], 4))

# 对比（MNN 内部可能做了数值优化，用宽松容差）
diff = np.abs(ort_out - mnn_out).max()
print(f"\nMNN vs ORT 最大差异: {diff:.6f}")
if diff < 1e-4:
    print("✅ MNN 和 ORT 结果一致！")
else:
    print("⚠️ 差异较大，检查一下")
