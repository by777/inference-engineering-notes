# 01_cross_target.py
# Lesson 18: TVM 交叉编译到 ARM 板子 - 实验 1: 交叉编译 target 语法
#
# 回顾:
#   Lesson 13-17: 所有实验都用 target="llvm"（生成 x86-64 机器码，本机直接跑）
#   本实验: 学习 TVM 怎么生成 aarch64 机器码——"交叉编译"的核心
#
# 核心概念: target 是"编译器要为哪台机器生成代码"的声明
#   - "llvm"                    → x86-64 本机（之前的实验）
#   - "llvm -mtriple=aarch64-linux-android" → ARM 板子（本实验）
#   - "opencl" / "cuda"         → GPU
#   - "hexagon"                 → Qualcomm Hexagon NPU（TVM 官方 target 之一）
#
# 交叉编译 vs 本机编译:
#   本机编译:  编译器生成 x86-64 代码 → 本机 CPU 直接跑（写代码的机器 = 跑代码的机器）
#   交叉编译:  编译器生成 aarch64 代码 → 复制到 ARM 板子跑（写代码的机器 ≠ 跑代码的机器）
#   TVM 的交叉编译 = 换一个 target 参数，就这么简单（LLVM 后端负责生成目标架构机器码）
#
# 运行: source .venv/bin/activate && python 01_cross_target.py

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
from tvm.script import tir as T


# -------- 1. 同一个内核，编译成两种架构 -------- #
# add_one: y[i] = x[i] + 1（实验 1 的内核，复习 TVMScript 三大件）
@T.prim_func
def add_one(
    x: T.Buffer(8, "float32"),
    y: T.Buffer(8, "float32"),
):
    for i in T.serial(8):
        y[i] = x[i] + 1.0


# 本机编译: target = "llvm" → 生成 x86-64 机器码
print("=" * 60)
print("本机编译 (x86-64): target = 'llvm'")
print("=" * 60)
f_x86 = tvm.build(add_one, target="llvm")
print("编译完成:", f_x86)

# 交叉编译: target = "llvm -mtriple=aarch64-linux-android"
#   -mtriple: 告诉 LLVM 生成什么架构的代码
#   aarch64 = ARM 64 位（板子 CPU 架构）
#   linux-android = 操作系统 + ABI（Android 用 Bionic libc）
print()
print("=" * 60)
print("交叉编译 (aarch64): target = 'llvm -mtriple=aarch64-linux-android'")
print("=" * 60)
f_arm = tvm.build(add_one, target="llvm -mtriple=aarch64-linux-android")
print("编译完成:", f_arm)


# -------- 2. 对比两种架构的汇编 -------- #
# 同一个 Python 内核 → 两种完全不同的机器码 → 这就是编译器的价值
print()
print("=" * 60)
print("x86-64 汇编（本机）")
print("=" * 60)
print(f_x86.get_source("asm")[:600])

print()
print("=" * 60)
print("aarch64 汇编（板子）")
print("=" * 60)
print(f_arm.get_source("asm")[:600])
# 关键差异:
#   x86-64:   pushq/movl/cmpl（AT&T 语法），寄存器 r*/e*
#   aarch64:  str/cmp/b.ne/cbz（ARM 指令），寄存器 w*/x*


# -------- 3. 验证: 交叉编译的内核在 x86 上跑不了！ -------- #
# 这是"交叉编译存在意义"的关键证明
print()
print("=" * 60)
print("关键验证: aarch64 代码不能在 x86 上直接执行")
print("=" * 60)
x_np = np.arange(8, dtype="float32")
y_tvm = tvm.nd.empty((8,), dtype="float32")
try:
    f_arm(tvm.nd.array(x_np), y_tvm)
    print("❌ 意外: aarch64 代码竟然在 x86 上跑通了？")
except Exception as e:
    print("✅ 如预期报错: 架构不匹配")
    print("   ", str(e)[:100], "...")
print()
print("解释: f_arm 生成的是 aarch64 机器码，x86 CPU 不认识这些指令")
print("      → 所以必须把 .so 推到 ARM 板子上才能执行")
print("      → 这就是'交叉编译'存在的意义: 在 x86 上生成板子能跑的代码")
