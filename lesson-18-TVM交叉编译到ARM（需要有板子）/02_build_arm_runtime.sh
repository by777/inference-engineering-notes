#!/bin/bash
# 02_build_arm_runtime.sh
# Lesson 18: TVM 交叉编译到 ARM 板子 - 实验 2: 用 NDK 编译 arm64 的 libtvm_runtime.so
#
# 【为什么要做这一步？】
#   runtime (libtvm_runtime.so) 是 TVM 的"执行环境"：负责加载内核 .so、管理内存、调度线程。
#   就像 Python 解释器之于 .py 文件——没有它，编译好的内核在板子上跑不起来。
#   本机 tvm-bin/ 里的 libtvm_runtime.so 是 x86 机器码，ARM 板子的 CPU 不认识。
#   → 必须用 NDK 的交叉编译器，把 runtime 编译成 arm64 版本。
#
# 【用法】
#   bash 02_build_arm_runtime.sh          # 用默认 NDK 路径
#   NDK=/opt/ndk bash 02_build_arm_runtime.sh   # 覆盖 NDK 路径
#
# 【产出】
#   tvm-bin-arm/libtvm_runtime.so  (arm64 架构)

set -e
# set -e 的作用: "出错即停"。
# 脚本里任何一条命令返回非 0（失败），立即终止整个脚本。
# 好处: 不会在编译失败后继续往下跑（避免把半成品 .so 当成成功产物）。
# 坏处: 出错时没有中间提示——所以脚本里重要步骤都加了 echo。

# ============================================================
# 可配置项（换环境改这里就行）
# ============================================================
# ${VAR:-默认值} 的语法: 如果环境变量 VAR 已设置则用它，否则用默认值。
# 这样既支持"环境变量覆盖"，又保证"不设置也能跑"。
# 例: NDK=/opt/ndk bash 02_build_arm_runtime.sh  → 用 /opt/ndk
#     bash 02_build_arm_runtime.sh              → 用默认路径
NDK="${NDK:-$HOME/toolchain/android-ndk-r29}"   # NDK 根目录（交叉编译器在这）
API="${API:-33}"                                       # Android API 级别（板子 Android 13 → 33）
# API 级别 = Android 系统的版本号:
#   Android 13 = API 33（你板子的版本）
#   编译器会按这个版本的 Bionic libc（Android 的 C 库）生成代码

# ============================================================
# 由脚本位置推导的路径（不用手改）
# ============================================================
# BASH_SOURCE[0] = 本脚本的文件名
# dirname        = 取所在目录
# 这样脚本挪到哪都能跑，不依赖绝对路径写死
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 本脚本所在目录
REPO="$(dirname "$SCRIPT_DIR")"                              # 上一级 = onnx_runtime_ops/
TVM_SRC="$REPO/tvm-src"                    # TVM 源码根目录
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64"       # NDK 里的编译工具链
BUILD_DIR="$TVM_SRC/build_android"         # 交叉编译的构建目录（和 x86 的 build 分开）
OUT_DIR="$REPO/tvm-bin-arm"                # arm64 产物放这里（和 x86 的 tvm-bin 区分）

# ============================================================
# 编译器路径
# ============================================================
# aarch64 = ARM 64 位架构（板子 CPU）
# android33 = 按 Android 13 (API 33) 的 Bionic libc 编译
# clang 是 NDK 的编译器（Android 官方编译器，基于 LLVM）
CC="$TOOLCHAIN/bin/aarch64-linux-android${API}-clang"      # C 编译器
CXX="$TOOLCHAIN/bin/aarch64-linux-android${API}-clang++"   # C++ 编译器
SYSROOT="$TOOLCHAIN/sysroot"     # 系统根目录：Android 的头文件 + 库（Bionic libc）
# SYSROOT 的作用: 告诉编译器"系统头文件和标准库在哪"。
# 交叉编译时，本机的 /usr/include 是 Linux 的，不是 Android 的。
# 必须指定 Android 的 sysroot，编译器才知道 Bionic libc 的接口。

# 工具链存在性检查（提前失败，而不是编译到一半才挂）
if [ ! -f "$CC" ]; then
    echo "❌ 找不到编译器: $CC"
    echo "   请检查 NDK 路径（$NDK）或 API 级别（$API）"
    exit 1   # 非 0 退出 = 失败（配合 set -e，脚本终止）
fi

echo "编译器: $CC"

# ============================================================
# 1. CMake 配置（生成 Makefile）
# ============================================================
# CMake 是构建工具: 根据配置生成 Makefile，make 再用 Makefile 编译
# -D 参数 = 传给 CMake 的配置项
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"   # 在 build_android 里配置，避免污染 x86 的 build 目录

# ── 各配置项含义 ──
# CMAKE_C_COMPILER / CMAKE_CXX_COMPILER:
#   指定 C/C++ 编译器（用 NDK 的 aarch64 clang）
# CMAKE_SYSROOT:
#   指定 Android 系统头文件/库目录
# CMAKE_BUILD_TYPE=Release:
#   编译优化模式（-O3，Release 比 Debug 快但没法调试）
# USE_LLVM=OFF:
#   ⚠️ 关键。LLVM 是"生成代码的工具"（编译器的一部分），只在 x86 开发机用。
#   runtime 是"执行环境"，不需要 LLVM。关掉它，编译更快、依赖更少。
#   （如果开着，CMake 会去找 LLVM，而交叉编译环境里没有 → 报错）
# USE_GRAPH_EXECUTOR=ON:
#   开启图执行器。板子上跑推理需要 graph_executor 组件（加载图、调度算子）。
#   Lesson 15/16/17 的 C 端 set_input/run/get_output 就是它提供的。
# USE_LIBBACKTRACE=OFF:
#   ⚠️ 必须关。libbacktrace 在 configure 阶段会"运行"它编译的程序做测试，
#   但交叉编译出的 arm64 程序在 x86 上跑不了 → 报错。
#   关掉后 TVM 不用 libbacktrace（只是调试栈回溯功能，不影响推理）。
# USE_RPC / USE_MICRO / USE_CPP_RPC = OFF:
#   关掉不需要的远程调用/微控制器功能（板子用不到，关掉减少编译量）
cmake -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DCMAKE_SYSROOT="$SYSROOT" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_LLVM=OFF \
      -DUSE_GRAPH_EXECUTOR=ON \
      -DUSE_LIBBACKTRACE=OFF \
      -DUSE_RPC=OFF \
      -DUSE_MICRO=OFF \
      -DUSE_CPP_RPC=OFF \
      "$TVM_SRC"

# ============================================================
# 2. 编译
# ============================================================
# make -j: 并行编译，$(nproc) = CPU 核数，全核并行加快速度
# 目标 tvm_runtime: 只编译 runtime，不编译整个编译器（libtvm.so）
#   为什么只编 runtime? 板子只需要"执行"编译好的内核，不需要"编译"。
#   编译器（libtvm.so）留在 x86 开发机用。
echo "=== 2. 编译 runtime ==="
make -j$(nproc) tvm_runtime

# ============================================================
# 3. 拷贝产物
# ============================================================
echo "=== 3. 拷贝产物 ==="
mkdir -p "$OUT_DIR"
cp "$BUILD_DIR/libtvm_runtime.so" "$OUT_DIR/"
echo "✅ 完成: $OUT_DIR/libtvm_runtime.so"

# file 命令显示文件架构，验证编译成功（应显示 ARM aarch64）
file "$OUT_DIR/libtvm_runtime.so"