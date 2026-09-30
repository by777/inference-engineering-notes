# 03_tvmscript_deploy.py
# Lesson 17: TVMScript - 实验 3: 把 MatMul 内核编译成 .so 给 C 调用
#
# 回顾:
#   Lesson 15/16: mlp3.onnx → relay.build → libmlp3.so → C/C++ 调用
#   实验 1/2:     TVMScript 手写内核 → tvm.build → Python 里直接调用
#   本实验:       MatMul → relay.build → libmatmul.so → C 程序调用
#
# 完整链路: 算子表达式 → TVM 编译成机器码 → 动态库 → 部署到 C 程序
#           （这就是"编译器"产品的核心工作流）
#
# 运行: source .venv/bin/activate && python 03_tvmscript_deploy.py
# 然后: make && ./main

# ========== 环境配置（必须在 import tvm 之前）==========
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "tvm-src" / "python"))
sys.path.insert(0, str(_REPO / "tvm-bin"))
os.environ["LD_LIBRARY_PATH"] = (
    str(_REPO / "tvm-bin") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ["TVM_NUM_THREADS"] = "16"

import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor

M, N, K = 256, 256, 256


# -------- 1. 用 Relay 算子构造 IRModule --------
# 用 relay.nn.matmul: Relay 的 matmul 算子（语义等价于实验 2 的 MatMul）
#   - A: [M,K] placeholder
#   - B: [K,N] placeholder
#   - main(A, B) = matmul(A, B) → [M,N]
#
# 为什么不用实验 2 的 TVMScript 内核？
#   TVMScript 写的 PrimFunc 可以直接 tvm.build 编译，但导出 .so 给 C 用时
#   会段错误（弱符号初始化问题，见底部【踩坑记录】）。
#   所以这里改用 relay.nn.matmul → relay.build 路径，和 Lesson 15/16 完全一致，
#   导出的 .so 带 __tvm_dev_mblob → C 端加载后自动构造 graph_executor，稳跑。
#
# 对比实验 2: 那里我们手写循环 + 手写调度，这里 Relay 自动生成最优代码
A = relay.var("A", shape=(M, K), dtype="float32")
B = relay.var("B", shape=(K, N), dtype="float32")
C = relay.nn.matmul(A, B)
func = relay.Function([A, B], C)
#      ↑           ↑      ↑
#    函数    输入列表   输出表达式
# 等价于 ONNX 图的 graph: 输入是 [A, B]，输出是 C
# 等价于 te 的: C = te.compute(..., name="C")
# 等价于 C 语言: float* matmul(float* A, float* B) { return C; }
# relay.nn.matmul(A, B) 不是 te.compute(...) 的直接替换。
# 它俩层级不同：一个是你手写的低层计算表达式，
# 一个是 TVM 预置的高层算子。
# relay.nn.matmul 内部隐含了一个 te.compute 实现，
# 但你是不能直接看到/改它的调度的。
mod = tvm.IRModule.from_expr(func)
# 把上面这个函数"打包"成一个编译器能处理的模块
# 等价于: 告诉编译器"这是我编译的入口 main"
# 编译器找入口函数时会找名叫 "main" 的（和 C 程序找 main 一样）
mod = relay.transform.InferType()(mod)  # 类型推断，让 Relay 知道 C 的形状

print("Relay IRModule:")
print(mod)
# 预期输出:
#   def @main(%A: Tensor[(256, 256), float32], %B: Tensor[(256, 256), float32])
#     -> Tensor[(256, 256), float32] {
#     nn.matmul(%A, %B, units=None)
#   }


# -------- 2. 用 relay.build 编译 + 导出 .so --------
# 和 Lesson 15/16 完全一致的流程：
#   relay.build: Relay IR → (pass 优化) → TIR → LLVM 机器码
#   export_library: 把编译产物 + graph_executor 元数据打包成 .so
target = tvm.target.Target("llvm")
with tvm.transform.PassContext(opt_level=3):  # 最高优化级别
    lib = relay.build(mod, target=target, params={})
print("编译完成:", lib)

lib.export_library("./libmatmul.so")
print("已导出 libmatmul.so")


# -------- 3. Python 侧自测（确保 .so 正确）--------
A_np = np.random.rand(M, K).astype("float32")
B_np = np.random.rand(K, N).astype("float32")
golden = A_np @ B_np

# 用 graph_executor 加载 .so 验证（和 C 端调用路径完全一致）
loaded_lib = tvm.runtime.load_module("./libmatmul.so")
dev = tvm.cpu(0)
m = graph_executor.GraphModule(loaded_lib["default"](dev))
m.set_input("A", tvm.nd.array(A_np))
m.set_input("B", tvm.nd.array(B_np))
m.run()
C_out = m.get_output(0).numpy()
np.testing.assert_allclose(C_out, golden, rtol=1e-4, atol=1e-4)
print("✅ Python 侧自测通过: graph_executor 路径数值正确")
print("→ C 端走 default 工厂 + set_input/run/get_output 三步（和 Lesson 16 一致）")


# -------- 4. 打印 Relay 生成的 TIR（可选，看编译器怎么调度）--------
# 注: relay.build 内部会调用 AutoScheduler / MetaSchedule 做自动调优
# 如果你想看 Relay 生成的 TIR 代码，可以取消下面注释：
# tir_mod = lib.get_executor().get_tir()  # 或 lib.get_tir_module()
# print(tir_mod)
