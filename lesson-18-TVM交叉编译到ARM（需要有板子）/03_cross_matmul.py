# 03_cross_matmul.py
# Lesson 18: TVM 交叉编译到 ARM 板子 - 实验 3: MatMul 交叉编译成 arm64 .so
#
# 回顾:
#   实验 1: 学会 target="llvm -mtriple=aarch64-linux-android" 生成 aarch64 机器码
#   实验 2: 用 NDK 编译了 arm64 版 libtvm_runtime.so（板子的执行环境）
#   实验 3: 把 MatMul 内核交叉编译 → C 程序 → 推到板子跑（完整闭环）
#
# 和 Lesson 17 实验 3 的对比:
#   Lesson 17: relay.build(..., target="llvm") → x86 .so → 本机 C 调用
#   本实验:    relay.build(..., target="llvm -mtriple=aarch64-linux-android")
#             → arm64 .so → 板子 C 调用
#   → 唯一的区别就是 target 参数！这就是交叉编译。
#
# 运行: source .venv/bin/activate && python 03_cross_matmul.py
# 产出: libmatmul_arm.so (arm64) + main_arm (arm64 可执行文件)

import os
import subprocess
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

M, N, K = 256, 256, 256

# ========== NDK 工具链路径（可配置，和实验 2 脚本保持一致）==========
NDK = os.environ.get("NDK", "$HOME/toolchain/android-ndk-r29")
API = os.environ.get("API", "33")
TOOLCHAIN = f"{NDK}/toolchains/llvm/prebuilt/linux-x86_64"
CROSS_CC = f"{TOOLCHAIN}/bin/aarch64-linux-android{API}-clang"

# ========== 1. 用 Relay 构造 MatMul（同 Lesson 17 实验 3）==========
A = relay.var("A", shape=(M, K), dtype="float32")
B = relay.var("B", shape=(K, N), dtype="float32")
C = relay.nn.matmul(A, B)
mod = tvm.IRModule.from_expr(relay.Function([A, B], C))
mod = relay.transform.InferType()(mod)

# ========== 2. 交叉编译: 唯一区别 = target 参数 ==========
target = tvm.target.Target("llvm -mtriple=aarch64-linux-android")
print("交叉编译 target:", target)
print("（和 Lesson 17 唯一的不同就是这里！）\n")

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params={})


# ⚠️ 关键: export_library 默认用系统 g++ 链接（x86）→ 链接不了 arm64 的 .o
#    必须传 fcompile 参数，指定用 NDK 的 aarch64 clang 来链接
#    这就是交叉编译的"最后一公里"：不仅代码生成要交叉，链接也要交叉
def cross_compile(output, inputs, options=None):
    """用 NDK aarch64 clang 链接 arm64 共享库"""
    cmd = [CROSS_CC, "-shared", "-fPIC", "-o", output] + list(inputs)
    if options:
        cmd += options
    print("链接命令:", " ".join(cmd))
    subprocess.run(cmd, check=True)


lib.export_library("./libmatmul_arm.so", fcompile=cross_compile)
print("✅ 已导出 arm64 版 libmatmul_arm.so")

# 验证 .so 架构
out = subprocess.run(["file", "libmatmul_arm.so"], capture_output=True, text=True)
print(out.stdout.strip())

# ========== 3. 用 NDK clang 编译 C 端程序 ==========
print("\n编译 C 端 main_arm (arm64)...")
tvm_src_include = str(_REPO / "tvm-src" / "include")
dlpack_include = str(_REPO / "tvm-src" / "3rdparty" / "dlpack" / "include")
dmlc_include = str(_REPO / "tvm-src" / "3rdparty" / "dmlc-core" / "include")
tvm_bin_arm = str(_REPO / "tvm-bin-arm")

# 编译 C 程序（arm64）
# ⚠️ Android/Bionic libc 自带 pthread，不需要 -lpthread（NDK 没有这个库名）
#    而 Linux/glibc 需要。这是 Android 和 Linux 平台的一个差异。
# ⚠️ -fPIE -pie: Android 可执行文件必须 PIE，否则无法执行
cmd = [
    CROSS_CC,  # NDK 的 arm64 clang（不是系统 gcc！）
    "-std=c17",  # C17 标准
    f"-I{tvm_src_include}",  # 头文件路径（dlpack 等）
    f"-I{dlpack_include}",
    f"-I{dmlc_include}",
    "-o",  # 输出可执行文件 ← 源文件
    "main_arm",
    "main_arm.c",
    f"-L{tvm_bin_arm}",  # 链接 arm64 的 runtime
    "-ltvm_runtime",
    "-ldl",  # 动态加载库（dlopen 需要）
    "-fPIE",  # Android 可执行文件必须 PIE
    "-pie",
]
print("编译命令:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("✅ 编译 main_arm 成功")

# 验证 C 程序架构
out = subprocess.run(["file", "main_arm"], capture_output=True, text=True)
print(out.stdout.strip())

print("""
════════════════════════════════════════════════
下一步（实验 4）: 推到板子运行
  adb shell mkdir -p /data/local/tmp/tvm
  adb push libmatmul_arm.so /data/local/tmp/tvm/
  adb push main_arm        /data/local/tmp/tvm/
  adb push libtvm_runtime.so /data/local/tmp/tvm/
  adb shell "cp /data/local/tmp/libc++_shared.so /data/local/tmp/tvm/ 2>/dev/null || true"
  adb shell "cd /data/local/tmp/tvm && LD_LIBRARY_PATH=. ./main_arm"
════════════════════════════════════════════════
""")
