# test_model.py
# Lesson 03: 测试带属性的自定义算子

import os
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, "libmy_ops.so")
MODEL_PATH = os.path.join(SCRIPT_DIR, "my_add.onnx")

sess_opts = ort.SessionOptions()
sess_opts.register_custom_ops_library(LIB_PATH)
sess = ort.InferenceSession(MODEL_PATH, sess_opts, providers=["CPUExecutionProvider"])
x = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float32)
y = np.array([6.0, 12.0, 18.0, 24.0], dtype=np.float32)
z = sess.run(["Z"], {"X": x, "Y": y})[0]
# 期望: (x + y) * scale = [11, 22, 33, 44] * 2.5
expected = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32) * 2.5
print(f"MyAdd result:  {z}")
print(f"  期望:          {expected}")
print(f"  状态:          {'✅' if np.allclose(z, expected) else '❌'}")
