# 02_tvm_arm_matmul_benchmark.py
# Lesson 20: 4 个调度在 ARM 板子上实测
#
# 回顾:
#   Lesson 17: 4 个调度在 x86 本机跑, opt_parallel 37.7x
#   实验 1:   TVM 生成 f32/int8 NEON 机器码, 反汇编对照 MNN
#   本实验:  同一份 4 个调度, 交叉编译到 ARM64, 推板子实测
#
# 运行: source .venv/bin/activate && python 02_tvm_arm_matmul_benchmark.py
# 产出: libmatmul_{naive,opt,parallel,opt_parallel}.so (arm64)

# ========== 环境配置（必须在 import tvm 之前）==========
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

import numpy as np
import tvm
from tvm.script import tir as T

M, N, K = 256, 256, 256


# ---------------- 1. naive: 三层串行循环 ----------------
@T.prim_func
def matmul_naive(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    for i, j in T.grid(M, N):
        C[i, j] = T.float32(0)
    for i, j, k in T.grid(M, N, K):
        C[i, j] += A[i, k] * B[k, j]


# ---------------- 2. opt: 循环重排 + 向量化 ----------------
@T.prim_func
def matmul_opt(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    for i in T.serial(M):
        for j in T.vectorized(N):
            C[i, j] = T.float32(0)
    for i in T.serial(M):
        for k in T.serial(K):
            for j in T.vectorized(N):
                C[i, j] += A[i, k] * B[k, j]


# ---------------- 3. parallel: 多核并行 (k 仍最内) ----------------
@T.prim_func
def matmul_parallel(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    for i in T.parallel(M):
        for j in T.vectorized(N):
            C[i, j] = T.float32(0)
    for i in T.parallel(M):
        for j in T.serial(N):
            for k in T.serial(K):
                C[i, j] += A[i, k] * B[k, j]


# ---------------- 4. opt_parallel: 三刀全上 ----------------
@T.prim_func
def matmul_opt_parallel(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    for i in T.parallel(M):
        for j in T.vectorized(N):
            C[i, j] = T.float32(0)
    for i in T.parallel(M):
        for k in T.serial(K):
            for j in T.vectorized(N):
                C[i, j] += A[i, k] * B[k, j]


# ========== 编译 4 个版本到 ARM64 ==========
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon")
print("[1] ARM target:", target)

# 用 NDK clang++ 做 fcompile, 否则 .so 带 glibc 依赖, 板子加载不了
NDK = os.environ.get("NDK", "$HOME/toolchain/android-ndk-r27d")
API = os.environ.get("API", "33")
CROSS_CXX = f"{NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android{API}-clang++"


def cross_compile(output, inputs, options=None):
    cmd = [CROSS_CXX, "-shared", "-fPIC", "-o", output] + list(inputs)
    if options:
        cmd += options
    subprocess.run(cmd, check=True)


funcs = {
    "naive": matmul_naive,
    "opt": matmul_opt,
    "parallel": matmul_parallel,
    "opt_parallel": matmul_opt_parallel,
}

OUT_DIR = Path(__file__).resolve().parent
print("[2] 4 个版本编译到 ARM64 ...")
for name, func in funcs.items():
    # name="default" 让 C 端统一用 TVMModGetFunction(mod, "default")
    f = tvm.build(func, target=target, name="default")
    so_path = OUT_DIR / f"libmatmul_{name}.so"
    f.export_library(str(so_path), fcompile=cross_compile)

    # 反汇编统计 NEON 指令数
    res = subprocess.run(
        ["aarch64-linux-gnu-objdump", "-d", str(so_path)],
        capture_output=True,
        text=True,
    )
    n_neon = sum(
        1
        for l in res.stdout.splitlines()
        if any(k in l.lower() for k in ["fmla", "ld1", "st1", "fmul"])
    )

    # 验证依赖是 Bionic 而非 glibc
    res_needed = subprocess.run(
        ["aarch64-linux-gnu-objdump", "-p", str(so_path)],
        capture_output=True,
        text=True,
    )
    needed_line = next(
        (l for l in res_needed.stdout.splitlines() if l.strip().startswith("NEEDED")),
        "无 NEEDED",
    )
    print(
        f"    {name:14s} {so_path.stat().st_size:6d} bytes  NEON={n_neon:3d}  {needed_line.strip()}"
    )

print("[完成] 主机准备完毕。下一步: make 编译 C 端, make run 推板子跑。")
