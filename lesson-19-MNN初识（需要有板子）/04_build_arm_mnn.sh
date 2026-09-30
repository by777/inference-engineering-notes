#!/bin/sh
# 04_build_arm_mnn.sh
# 用 NDK 交叉编译 arm64 版 libMNN.so（板子上跑 MNN 推理的引擎）
#
# 用法: sh 04_build_arm_mnn.sh
# 产出: mnn-src/build_android/OFF/arm64-v8a/libMNN.so
# 前置: 在 lesson-19-MNN初识 目录里运行

set -e

# ── 可配置项（换环境改这里）──
NDK="${NDK:-$HOME/toolchain/android-ndk-r27d}"  # 标准版 NDK（自带 toolchain cmake）
API="${API:-33}"                                       # Android 13 = API 33

# ── 路径推导（用 $0，任意目录运行都行）──
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$SCRIPT_DIR")
MNN_SRC="$REPO/mnn-src"
BUILD_DIR="$MNN_SRC/build_android"

# ── 检查 toolchain 文件存在 ──
TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"
[ -f "$TOOLCHAIN" ] || { echo "❌ 找不到 $TOOLCHAIN"; exit 1; }

echo "=== 1. 配置 CMake ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CMAKE_TOOLCHAIN_FILE = NDK 自带的交叉编译配置，一个文件搞定:
#   自动设置目标平台(Android/aarch64)、编译器、汇编器、sysroot、API 级别
#   不需要手动写 CMAKE_SYSTEM_NAME / CMAKE_SYSTEM_PROCESSOR 等一堆参数
#
# ANDROID_ABI=arm64-v8a     目标是 64 位 ARM
# ANDROID_PLATFORM=android-33  对应 Android 13 的 Bionic libc
# MNN_BUILD_CONVERTER=OFF  板子不需要模型转换器（转换在 PC 上做）
# MNN_USE_SSE=OFF           SSE 是 x86 指令集，ARM 上必须关
# MNN_BUILD_TRAIN/TEST/BENCHMARK=OFF  板子只跑推理，关掉这些省编译时间
cmake "$MNN_SRC" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-$API \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_BUILD_CONVERTER=OFF \
    -DMNN_BUILD_TRAIN=OFF \
    -DMNN_BUILD_TEST=OFF \
    -DMNN_BUILD_BENCHMARK=OFF \
    -DMNN_USE_SSE=OFF

echo "=== 2. 编译 ==="
make -j$(nproc)

echo "=== 3. 验证 ==="
# MNN 把 OFF 开关的产物放 OFF/arm64-v8a/ 子目录
LIBMNN="$BUILD_DIR/OFF/arm64-v8a/libMNN.so"
file "$LIBMNN"
echo "（应显示 ARM aarch64）"