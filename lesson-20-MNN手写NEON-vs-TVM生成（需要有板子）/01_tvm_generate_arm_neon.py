# 01_tvm_generate_arm_neon.py
# Lesson 20: MNN 手写 NEON vs TVM 生成 - 实验 1: TVM 编译 ARM64 + 反汇编
#
# 回顾:
#   Lesson 17: TVMScript 写 MatMul 4 调度 → 本机 x86 跑 → opt_parallel 37.7x
#   Lesson 18: 交叉编译到 aarch64 → 推到 某 arm64 开发板 板子跑通
#   Lesson 19: MNN 用 source/backend/cpu/arm/arm64/ 下的 .S 手写汇编
#   本实验: 让 TVM 编译出 ARM64 NEON 机器码，反汇编后对照 MNN 手写 NEON
#
# 核心问题：
#   1. TVM 通过 T.vectorized(N) 让 LLVM ARM64 后端生成什么 NEON 指令？
#   2. TVM 自动生成的代码 vs MNN 手写 NEON，差异在哪儿？
#   3. TVM 对 int8 matmul 能自动生成 smull/sadalp 16x4 tiling 吗？
#
# 运行: source .venv/bin/activate && python 01_tvm_generate_arm_neon.py

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

# ============================================================
# 1. TVMScript: float32 matmul（同 Lesson 17 opt 版，缩小到 128×128）
# ============================================================
M, N, K = 128, 128, 128


# TVMScript 定义 f32 matmul 只写了"调度意图"（serial/vectorized），
# 具体生成什么 NEON 指令完全交给 LLVM 后端。这就是"编译器路径"的起点。
@T.prim_func
def matmul_neon(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    # 循环重排（k 中间） + j 向量化
    # → LLVM ARM64 后端应该生成 NEON fmla 指令
    for i in T.serial(M):
        for j in T.vectorized(N):
            C[i, j] = T.float32(0)
    for i in T.serial(M):
        for k in T.serial(K):
            for j in T.vectorized(N):
                C[i, j] += A[i, k] * B[k, j]


# ============================================================
# 2. 编译为 ARM64 .so（用 NDK clang++ 链接，Bionic libc 兼容）
# ============================================================
# ⚠️ 踩坑 1: target 不能写 --host=arm-linux-gnu（不是合法 target kind）
#    正确写法: 只用 -mtriple + -mattr，host 默认 = target

# llvm                       -mtriple=aarch64-linux-gnu            -mattr=+neon
# └─ 后端：用 LLVM 生成代码    └─ 目标架构：ARM 64 位 Linux          └─ 特性：启用 NEON
# -mattr = machine attributes（LLVM 的 CPU 特性开关参数）
# + = 打开（- 号则是关闭，如 -mattr=-neon）
# neon = NEON——ARM 的 128-bit SIMD 向量指令扩展（Advanced SIMD）
target_arm = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+neon")
print(f"[1] ARM target: {target_arm.kind} / {target_arm.attrs}")

print("[2] 编译 float32 matmul 为 ARM64 ...")
# ⚠️ 踩坑 2: tvm.lower(func, target=...) 在 0.18.0 没有 target 参数
#    直接用 tvm.build
f_arm = tvm.build(matmul_neon, target=target_arm, name="matmul_neon")

# ⚠️ 踩坑 3: TVM 默认 export_library 用 x86 g++ 链接
#    对 ARM64 .o 报错 "Relocations in generic ELF (EM: 183)"
#    → 必须用交叉链接器
# ⚠️ 踩坑 4: ABI 一致性铁律
#    主可执行程序 / libtvm_runtime.so / matmul .so 必须都用 NDK Bionic ABI
#    不能跨 NDK clang++ × aarch64-linux-gnu-g++ 混编（symbol mangling 不兼容）
# 用 NDK r27d 的 clang++ 链接到 Bionic libc（Android 兼容）
NDK = os.environ.get("NDK", "$HOME/toolchain/android-ndk-r27d")
API = os.environ.get("API", "33")
CROSS_CXX = (
    f"{NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/"
    f"aarch64-linux-android{API}-clang++"
)


def cross_compile(output, inputs, options=None):
    """用 NDK clang++ 链接 ARM .so → Bionic libc 兼容"""
    cmd = [CROSS_CXX, "-shared", "-fPIC", "-o", output] + list(inputs)
    if options:
        cmd += options
    subprocess.run(cmd, check=True)


so_path = Path(__file__).resolve().parent / "libmatmul_arm.so"
f_arm.export_library(str(so_path), fcompile=cross_compile)
print(f"    .so 大小: {so_path.stat().st_size} bytes")

# 验证 .so 架构 + 依赖
out = subprocess.run(["file", str(so_path)], capture_output=True, text=True)
print(f"    架构: {out.stdout.strip()}")
out = subprocess.run(
    ["aarch64-linux-gnu-objdump", "-p", str(so_path)], capture_output=True, text=True
)
needed = [l for l in out.stdout.splitlines() if "NEEDED" in l]
print(f"    依赖: {needed}")

# ============================================================
# 3. 反汇编 .so，提取 TVM 生成的 ARM64 NEON 指令
# ============================================================
print("\n[3] 反汇编（aarch64-linux-gnu-objdump -d）看 NEON 指令...")
res = subprocess.run(
    ["aarch64-linux-gnu-objdump", "-d", str(so_path)], capture_output=True, text=True
)
disasm = res.stdout

# NEON 浮点指令关键字
#   fmla  = fused multiply-accumulate（融合乘加，对应 A[i,k] * B[k,j] += C[i,j]）
#   ld1   = 向量加载（一次 16B = 4 个 f32）
#   st1   = 向量存储
#   fmul  = 向量乘
#   fmov  = 标量 ↔ 向量 lane 搬运
neon_keywords = [
    "ld1",
    "st1",
    "fmla",
    "fmlal",
    "fmul",
    "fadd",
    "fmov",
    "fcvt",
    "scvtf",
    "fmin",
    "fmax",
    "dup",
]
neon_lines = [
    line.rstrip()
    for line in disasm.splitlines()
    if any(k in line.lower() for k in neon_keywords)
]
print(f"    共 {len(neon_lines)} 行含 NEON 指令")
print("    前 20 行示例:")
for l in neon_lines[:20]:
    print(f"      {l}")

# 保存完整反汇编供对照
disasm_path = Path(__file__).resolve().parent / "tvm_matmul_arm.disasm"
disasm_path.write_text(disasm)
neon_path = Path(__file__).resolve().parent / "tvm_matmul_arm_neon_only.txt"
neon_path.write_text("\n".join(neon_lines))
print(f"\n[4] 完整反汇编: {disasm_path}")
print(f"    NEON-only 摘录: {neon_path}")

# ============================================================
# 4. 额外实验: int8 直接 matmul（看 TVM 能否生成 smull/sadalp）
# ============================================================
# 为什么测 int8？因为 MNN 的 MNNGemmInt8AddBiasScale_16x4_Unit.S 全是手写
# smull + smlal2 + sadalp 的 pipeline。如果 TVM 对裸 int8 matmul 能自动生成
# 同样的指令组合，说明 LLVM 后端已经够强，MNN 手写汇编就没必要了。
# 结果出乎意料——见后面的输出。
print("\n[5] 额外实验: int8 matmul（看 TVM 能否自动生成 smull/sadalp）")


@T.prim_func
def matmul_int8(
    A: T.Buffer((M, K), "int8"),
    B: T.Buffer((K, N), "int8"),
    C: T.Buffer((M, N), "int32"),
):
    # 把 K 拆成外 8 + 内 8，内层尝试 vectorized
    for i in T.serial(M):
        for j in T.serial(N):
            for ko in T.serial(K // 8):
                for ki in T.vectorized(8):
                    C[i, j] += A[i, ko * 8 + ki].astype("int32") * B[
                        ko * 8 + ki, j
                    ].astype("int32")


print("    编译 int8 matmul ...")
try:
    f_i8 = tvm.build(matmul_int8, target=target_arm, name="matmul_int8")
    so_i8_path = Path(__file__).resolve().parent / "libmatmul_int8_arm.so"
    f_i8.export_library(str(so_i8_path), fcompile=cross_compile)
    print(f"    int8 .so: {so_i8_path} ({so_i8_path.stat().st_size} bytes)")
except Exception as e:
    print(f"    int8 build 失败: {e}")
    so_i8_path = None

if so_i8_path and so_i8_path.exists():
    res = subprocess.run(
        ["aarch64-linux-gnu-objdump", "-d", str(so_i8_path)],
        capture_output=True,
        text=True,
    )
    i8_disasm = res.stdout
    # int8 NEON 关键指令
    #   smull  = 8 路 int8 × int8 → int16 乘（一次 8 个 MAC）
    #   smlal2 = 同上但取高 8 路，再加到 v8
    #   sadalp = pairwise 累加 int16 → int32
    #   sxtl   = sign-extend int8 → int16
    i8_keywords = [
        "smull",
        "smlal",
        "sadalp",
        "saddlp",
        "smlal2",
        "ld1",
        "st1",
        "sxtl",
    ]
    i8_neon = [
        line
        for line in i8_disasm.splitlines()
        if any(k in line.lower() for k in i8_keywords)
    ]
    print(f"    int8 NEON 指令: {len(i8_neon)} 行")
    for l in i8_neon[:15]:
        print(f"      {l}")
    (Path(__file__).resolve().parent / "tvm_matmul_int8.disasm").write_text(i8_disasm)
    (Path(__file__).resolve().parent / "tvm_matmul_int8_neon_only.txt").write_text(
        "\n".join(i8_neon)
    )
    print("    保存到 tvm_matmul_int8.disasm / tvm_matmul_int8_neon_only.txt")

# ============================================================
# 5. 结果对照 + 解读
# ============================================================
print("""
════════════════════════════════════════════════════════════════
实验 1 结果分析（对照 MNN 手写 NEON）
════════════════════════════════════════════════════════════════

float32 版（TVM 自动生成）:
  `fmla v?.4s, v?.4s, v0.s[0]`  × N
  → 一次 4-lane 融合乘加 + 标量广播
  → 用满 v0~v31 全部 32 个 NEON 寄存器（register tiling）
  → 但 LLVM 把 unroll 圈数选大 → 部分累加器溢出到栈（str q, [sp]）

int8 版（TVM 自动生成）:
  只 4 行 NEON: ld1r / sxtl / smull2
  → 没有 16x4 tile，没有多套 pipeline
  → 远不及 MNN 的 MNNGemmInt8AddBiasScale_16x4_Unit.S
    （MNN 手写 8 套 smull + 8 套 smlal2 + 8 套 sadalp 一次 8x8 矩阵 MAC）

关键洞察:
  LLVM 对 f32 SIMD 向量化能力强（autovectorizer + register allocator 联手）
  LLVM 对 int8 自动向量化能力弱（没有 16x4 这种 layout-specific 优化）
  → 这就是 MNN 必须手写 ARM64 汇编的根本原因
  → 手写汇编的本质收益: layout-tied 调度 + 量化算子 fusion

下一步实验 02: 在板子上实测 4 个调度（naive/opt/parallel/opt_parallel）的加速比
""")
