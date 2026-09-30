# test_model.py
# Lesson 05: 测试多输入多输出自定义算子

import os
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(SCRIPT_DIR, "libmy_ops.so")
MODEL_PATH = os.path.join(SCRIPT_DIR, "my_add_sub.onnx")

sess_opts = ort.SessionOptions()
sess_opts.register_custom_ops_library(LIB_PATH)

sess = ort.InferenceSession(
    MODEL_PATH,
    sess_opts,
    providers=["CPUExecutionProvider"],
)

x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

sum_out, diff_out = sess.run(["Sum", "Diff"], {"X": x, "Y": y})

expected_sum = x + y
expected_diff = x - y

sum_ok = np.allclose(sum_out, expected_sum)
diff_ok = np.allclose(diff_out, expected_diff)

print(f"Sum result :  {sum_out}")
print(f"Sum expected:  {expected_sum}")
print(f"Sum status :  {'✅' if sum_ok else '❌'}")

print(f"Diff result : {diff_out}")
print(f"Diff expected: {expected_diff}")
print(f"Diff status : {'✅' if diff_ok else '❌'}")

all_ok = sum_ok and diff_ok
print(f"Overall     : {'✅' if all_ok else '❌'}")
