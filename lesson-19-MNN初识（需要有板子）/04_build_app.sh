#!/bin/sh
# 04_build_app.sh
# 交叉编译 C++ 推理程序 → 推到板子 → 运行
#
# 用法: sh 04_build_app.sh
# 前置: 先跑 04_build_arm_mnn.sh 生成 libMNN.so
# 注意: 在 lesson-19-MNN初识 目录里运行

set -e

# ── 可配置项 ──
NDK="${NDK:-$HOME/toolchain/android-ndk-r27d}"
API="${API:-33}"
BOARD_DIR="/data/local/tmp/mnn"  # 板子上放文件的目录

# ── 路径推导 ──
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$SCRIPT_DIR")
MNN_SRC="$REPO/mnn-src"
CXX="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${API}-clang++"
LIBMNN="$MNN_SRC/build_android/OFF/arm64-v8a/libMNN.so"

echo "=== 1. 交叉编译推理程序 ==="
# -fPIE -pie: Android 可执行文件必须位置无关，否则起不来
# 不加 -lpthread: Android Bionic libc 自带 pthread（Linux glibc 才需要这个库名）
$CXX -std=c++11 \
    -I"$MNN_SRC/include" \
    -o mnn_cpp_arm "$SCRIPT_DIR/03_mnn_cpp.cpp" \
    -L"$(dirname "$LIBMNN")" -lMNN \
    -fPIE -pie

file mnn_cpp_arm
echo "（应显示 ARM aarch64 PIE）"

echo "=== 2. 推到板子 ==="
adb shell mkdir -p "$BOARD_DIR"
adb push mnn_cpp_arm "$BOARD_DIR/"         # 可执行程序
adb push "$SCRIPT_DIR/mlp3.mnn" "$BOARD_DIR/"  # 模型（数据文件，直接拷）
adb push "$LIBMNN" "$BOARD_DIR/"           # arm64 推理引擎

# libMNN.so 是 C++ 写的，运行时需要 C++ 标准库
# 板子上 /data/local/tmp/ 已有（KWS 项目留下的），复制过来
# 不用 2>/dev/null || true: 那会吞掉错误，复制失败看不见
if adb shell "[ -f /data/local/tmp/libc++_shared.so ]"; then
    adb shell "cp /data/local/tmp/libc++_shared.so $BOARD_DIR/"
    echo "已复制 libc++_shared.so"
else
    echo "⚠️ 板子上没有 libc++_shared.so"
fi

echo "=== 3. 板子上运行 ==="
# LD_LIBRARY_PATH=. : Android 链接器默认不搜当前目录，必须显式指定
adb shell "cd $BOARD_DIR && chmod +x mnn_cpp_arm && LD_LIBRARY_PATH=. ./mnn_cpp_arm"