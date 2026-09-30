#!/bin/bash
# 04_run_on_board.sh
# Lesson 18: TVM 交叉编译到 ARM 板子 - 实验 4: 推送 + 板子运行
#
# 把实验 3 编译好的 arm64 产物推到板子并运行:
#   libmatmul_arm.so   → MatMul 内核（arm64，实验 3 产物）
#   main_arm           → C 端可执行程序（arm64，实验 3 产物）
#   libtvm_runtime.so  → arm64 版 runtime（实验 2 产物）
#   libc++_shared.so   → C++ 标准库（板子 /data/local/tmp 里已有，复制过来）
#
# 用法: bash 04_run_on_board.sh
# 前置: 实验 3 已生成 libmatmul_arm.so + main_arm

set -e

# ========== 由脚本位置推导路径 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
BOARD_DIR="/data/local/tmp/tvm"        # 板子上放文件的目录

echo "=== 1. 创建板子目录 ==="
adb shell mkdir -p "$BOARD_DIR"

echo "=== 2. 推送 arm64 文件 ==="
adb push libmatmul_arm.so "$BOARD_DIR/"
adb push main_arm "$BOARD_DIR/"
adb push "$REPO/tvm-bin-arm/libtvm_runtime.so" "$BOARD_DIR/"
# libc++_shared.so 板子上 /data/local/tmp/ 已有，复制到工作目录即可
adb shell "cp /data/local/tmp/libc++_shared.so $BOARD_DIR/ 2>/dev/null || true"

echo "=== 3. 板子上运行 ==="
# LD_LIBRARY_PATH=. : 让 Android 链接器在当前目录找 .so
#（Android 默认只搜系统目录，不搜当前目录，所以必须显式指定）
adb shell "cd $BOARD_DIR && chmod +x main_arm && LD_LIBRARY_PATH=. ./main_arm"