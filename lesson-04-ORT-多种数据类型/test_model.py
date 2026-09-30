# test_model.py
# Lesson 04: 测试多种数据类型

import os
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, "libmy_ops.so")

# 测试数据
test_cases = [
    ("float", np.float32),
    ("double", np.float64),
    ("int32", np.int32),
    ("int64", np.int64),
]

for dtype_name, np_dtype in test_cases:
    MODEL_PATH = os.path.join(SCRIPT_DIR, f"my_add_{dtype_name}.onnx")

    sess_opts = ort.SessionOptions()
    sess_opts.register_custom_ops_library(LIB_PATH)
    sess = ort.InferenceSession(
        MODEL_PATH, sess_opts, providers=["CPUExecutionProvider"]
    )

    x = np.array([5.0, 10.0, 15.0, 20.0], dtype=np_dtype)
    y = np.array([6.0, 12.0, 18.0, 24.0], dtype=np_dtype)
    z = sess.run(["Z"], {"X": x, "Y": y})[0]

    scale = np.array(2.5, dtype=np_dtype)
    expected = (x + y) * scale
    match = np.allclose(z, expected)

    print(
        f"{dtype_name:8s} | result: {z} | expected: {expected} | {'✅' if match else '❌'}"
    )
