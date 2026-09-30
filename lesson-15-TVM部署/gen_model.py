# gen_model.py
# Lesson 15: TVM 部署 - 把模型编译成独立 .so（类似 ONNX Runtime 的优化模型文件）
#
# 流程: mlp3.onnx → Relay → 编译 → export_library() 导出 .so
# 产出: libmlp3.so （C/C++ 可以直接加载推理）
#
# 运行: python gen_model.py

# ========== 环境配置（必须在 import tvm 之前）==========
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent  # onnx_runtime_ops/
sys.path.insert(0, str(_REPO / "tvm-src" / "python"))
sys.path.insert(0, str(_REPO / "tvm-bin"))
os.environ["LD_LIBRARY_PATH"] = (
    str(_REPO / "tvm-bin") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
)

# ========== 正式 import ==========
import onnx
import tvm
from tvm import relay

# ----- 1. 加载onnx-> Relay ----- #
onnx_model = onnx.load("mlp3.onnx")
mod, params = relay.frontend.from_onnx(onnx_model)
target = "llvm"  # CPU

# ----- 2. 编译 Relay 模型 ----- #
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)


# ---------- 3. 导出成独立 .so ----------#
# export_library 把编译产物打包成 .so，C++ 程序可以直接 dlopen 加载
# 注意: 需要指定 libtvm_runtime 的位置（链接时用）
lib.export_library("libmlp3.so", cc="g++")
print("已导出: libmlp3.so")

# ----------- 4. 打印输入输出信息（c++ 调用时需要） -----------
# TVM 的 graph executor 把权重和真输入都叫"input node"（p0~p5 是权重，X 是真输入）
# 正确的区分方式: 从 Relay 主函数签名读——不在 params 字典里的才是真输入
from tvm.contrib import graph_executor

dev = tvm.cpu()
m = graph_executor.GraphModule(lib["default"](dev))

# --- 真输入: 从 Relay 主函数签名拿（权威来源）---
print("输入:")
for param in mod["main"].params:
    name = param.name_hint
    if name not in params:   # 排除权重，只打印真输入
        # 从 checked_type 拿 shape/dtype
        ttype = param.checked_type
        print(f"  {name}: shape={[d for d in ttype.shape]}, dtype={ttype.dtype}")

# --- 输出: 跑一次，从实际 tensor 反推（0.18 没有直接 API）---
import numpy as np
m.set_input("X", tvm.nd.array(np.zeros((1, 256), dtype="float32")))
m.run()
out = m.get_output(0)
print(f"输出: shape={list(out.shape)}, dtype={out.dtype}")
