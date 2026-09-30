#!/bin/bash
# C3_build.sh
# 实验C: 编译 + 运行 int8 执行链观察
#
# 编译: 用 g++ 把 C3_trace.cpp 链接 libMNN.so
#   -I../mnn-src/include        : MNN 头文件 (Interpreter.hpp / Tensor.hpp)
#   -L../mnn-src/build -lMNN    : 链接 libMNN.so
#   -o C3_trace                 : 输出可执行文件
#
# 运行: 需要 LD_LIBRARY_PATH 指向 libMNN.so, 否则运行时找不到动态库
#
# 用法: bash C3_build.sh

set -e
cd "$(dirname "$0")"

echo "==== 编译 C3_trace ===="
g++ -std=c++11 C3_trace.cpp \
    -I../mnn-src/include \
    -L../mnn-src/build -lMNN \
    -o C3_trace

echo "==== 运行 ===="
LD_LIBRARY_PATH=../mnn-src/build ./C3_trace