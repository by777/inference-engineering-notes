# 01_tvmscript_intro.py
# Lesson 17: TensorIR / TVMScript - 实验 1: TVMScript 基本语法
#
# 回顾 (Lesson 13): te.compute 声明式写算子 → 编译器自己生成循环
# 本课: TVMScript 直接用 Python 语法"写循环"（像写 C，但更简洁）
#
# TVMScript 三大件:
#   @T.prim_func    → 声明这是一个"底层计算函数"（TIR PrimFunc）
#   T.Buffer(...)   → 声明输入输出张量（形状 + 数据类型）
#   T.serial / T.grid / T.range → 写循环
#
# 与 te 的本质区别:
#   te: 你描述"算什么"（声明式），编译器决定"怎么算"
#   TVMScript: 你直接写"怎么算"（命令式），完全掌控循环/内存
#   两者都能编译成机器码，但 TVMScript 让你能看见并修改底层 IR
#
# 运行: source .venv/bin/activate && python 01_tvmscript_intro.py

# ========== 环境配置（必须在 import tvm 之前）==========
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent  # onnx_runtime_ops/
sys.path.insert(0, str(_REPO / "tvm-src" / "python"))  # tvm Python 包（源码编译）
sys.path.insert(0, str(_REPO / "tvm-bin"))  # libtvm.so / libtvm_runtime.so
os.environ["LD_LIBRARY_PATH"] = (
    str(_REPO / "tvm-bin") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
)
os.environ["TVM_NUM_THREADS"] = "16"  # 并行线程数

import numpy as np
import tvm
from tvm.script import tir as T  # T = TVMScript 方言


# -------- 1. 最简单的 TVMScript：y[i] = x[i] + 1 -------- #
# te 写法:  y = te.compute((n,), lambda i: x[i] + 1, name="y")
# TVMScript 写法: 直接写循环，像写 C 一样
@T.prim_func
def add_one(
    x: T.Buffer(8, "float32"),  # 输入：8 个 float32（T.Buffer(形状, 类型)）
    y: T.Buffer(8, "float32"),  # 输出：8 个 float32
):
    for i in T.serial(8):  # 串行循环 i = 0..7（等价于 range(8)）
        y[i] = x[i] + 1.0  # 逐元素 +1，这就是内核本体


# -------- 2. 编译并运行 -------- #
# tvm.build: PrimFunc → 可执行模块（LLVM 后端生成机器码）
f = tvm.build(add_one, target="llvm")
print("编译完成, f =", f)  # Module(llvm, ...)

# 构造输入（dtype 必须和 T.Buffer 声明一致: float32）
x_np = np.arange(8, dtype="float32")

# ⚠️ 输出必须用 tvm.nd.empty 创建！运行后 .numpy() 取回
# ❌ 错误: tvm.nd.array(y_np) —— 会拷贝 numpy 数据到新张量，
#    运行结果写进 TVM 张量，不会同步回 y_np，y_np 仍是垃圾值
# ✅ 正确: empty 创建空张量，跑完 .numpy() 把结果拷回 numpy
x_tvm = tvm.nd.array(x_np)  # numpy → TVM 张量（输入）
y_tvm = tvm.nd.empty((8,), dtype="float32")  # 分配输出空间
f(x_tvm, y_tvm)  # 直接像函数一样调用
y_np = y_tvm.numpy()  # TVM 张量 → numpy（取回结果）

print("输入 x =", x_np)
print("输出 y =", y_np)

# 验证：y[i] 应该等于 x[i] + 1
expected = x_np + 1.0
assert np.allclose(y_np, expected), "结果不对!"
print("✅ 验证通过: y = x + 1")


# -------- 3. 对比 te 写法（回顾 Lesson 13）-------- #
# te 是"声明式"：描述要算什么，编译器生成循环
# TVMScript 是"命令式"：直接写循环，完全掌控
# 编译产物等价，但 TVMScript 能看到/修改底层 IR
import tvm.te as te

n = 8
x_te = te.placeholder((n,), name="x")
y_te = te.compute((n,), lambda i: x_te[i] + 1.0, name="y")
# te 的产出其实是 PrimFunc —— 用 create_prim_func 显式转换后同样可 build
f_te = tvm.build(te.create_prim_func([x_te, y_te]), target="llvm")

y_te_tvm = tvm.nd.empty((8,), dtype="float32")  # 同样用 empty（原因同上）
f_te(tvm.nd.array(x_np), y_te_tvm)
y_te_np = y_te_tvm.numpy()
print("te 版输出 y =", y_te_np)
np.testing.assert_allclose(y_np, expected, rtol=1e-5, atol=1e-5)
print("✅ te 版本与 TVMScript 版本结果一致")


# -------- 4. 看看编译出的 IR（TVMScript 的核心优势）-------- #
# ① TIR IR：PrimFunc 内部的中间表示，可还原成 TVMScript 文本
print("\n=== 打印 add_one 的 TIR IR ===")
print(add_one.script())  # 看到编译器眼中的循环结构

# ② LLVM IR：落到 LLVM 后端后的中间表示（优化后）
# ⚠️ LLVM 后端不支持 get_source("c") —— C 源码需要 CCodeGen 后端
#    LLVM target 支持 "ll" (LLVM IR) 和 "asm" (汇编)
print("\n=== 打印 LLVM IR（优化后，前 2000 字符）===")
print(f.get_source("ll")[:2000])
# 能看到编译器自动生成:
#   - 参数类型断言（arg 必须是 DLTensor 指针、ndim==1、dtype==float32）
#   - 这段代码相当于 C API 的"函数签名检查"层
# 这印证了 Lesson 16: TVMFuncCall 传参时为什么有 type_code / shape 检查
