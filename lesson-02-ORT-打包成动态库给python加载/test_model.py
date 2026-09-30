# test_model.py
# Lesson 02: 用 Python 加载 libmy_ops.so 并跑推理
# 依赖: pip install onnxruntime>=1.27 numpy

import os
import numpy as np
import onnxruntime as ort

# 用绝对路径, 避免工作目录不在脚本目录时找不到文件
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, "libmy_ops.so")
MODEL_PATH = os.path.join(SCRIPT_DIR, "my_add.onnx")

# 1. 创建 SessionOptions
sess_opts = ort.SessionOptions()

# 2. 注册自定义算子库
#    ORT 会 dlopen LIB_PATH, 找到 RegisterCustomOps 符号并调用
sess_opts.register_custom_ops_library(LIB_PATH)

# 3. 创建 session
sess = ort.InferenceSession(MODEL_PATH, sess_opts, providers=["CPUExecutionProvider"])
# 下面这行会报错 : ./my_add.onnx failed:Fatal error: my_domain:MyAdd(-1) is not a registered function/op
# 因此同时需要onnx和so
# sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# 4. 准备输入
x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

# 5. 跑推理
z = sess.run(["Z"], {"X": x, "Y": y})[0]

print(f"MyAdd result: {z}")
print(
    f"  期望:       [11. 22. 33. 44.]   "
    f"{'✅' if np.allclose(z, [11, 22, 33, 44]) else '❌'}"
)
