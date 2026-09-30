# test_model.py
# Lesson 06: 测试带属性的 ReduceMean

import os
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, "libmy_ops.so")

x = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ],
    dtype=np.float32,
)

test_cases = [
    {
        "name": "axis0_keep0",
        "expected": np.array([2.5, 3.5, 4.5], dtype=np.float32),
        "shape": (3,),
    },
    {
        "name": "axis1_keep0",
        "expected": np.array([2.0, 5.0], dtype=np.float32),
        "shape": (2,),
    },
    {
        "name": "axis1_keep1",
        "expected": np.array([[2.0], [5.0]], dtype=np.float32),
        "shape": (2, 1),
    },
]

for case in test_cases:
    model_path = os.path.join(SCRIPT_DIR, f"my_reduce_mean_{case['name']}.onnx")

    sess_opts = ort.SessionOptions()
    sess_opts.register_custom_ops_library(LIB_PATH)

    sess = ort.InferenceSession(
        model_path,
        sess_opts,
        providers=["CPUExecutionProvider"],
    )

    y = sess.run(["Y"], {"X": x})[0]
    # 用 allclose 而不是 ==，因为浮点计算会有微小误差。
    value_ok = np.allclose(y, case["expected"])
    shape_ok = tuple(y.shape) == case["shape"]

    print(f"{case['name']}:")
    print(f"  result   = {y}")
    print(f"  expected = {case['expected']}")
    print(f"  shape    = {y.shape}, expected {case['shape']}")
    print(f"  value    = {'✅' if value_ok else '❌'}")
    print(f"  shape    = {'✅' if shape_ok else '❌'}")
    print(f"  overall  = {'✅' if (value_ok and shape_ok) else '❌'}")
    print()
