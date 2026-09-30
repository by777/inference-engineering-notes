#!/bin/bash
# B0_build.sh
# 实验B: 编译 + 运行 PluginScale 自定义算子
# 用法: bash B0_build.sh

set -e
cd "$(dirname "$0")"

MNN=../mnn-src

echo "==== 编译 B0_plugin_scale ===="
g++ -std=c++11 B0_plugin_scale.cpp \
    -I$MNN/include \
    -I$MNN/schema/current \
    -I$MNN/3rd_party/flatbuffers/include \
    -L$MNN/build -lMNN \
    -L$MNN/build/express -lMNN_Express \
    -o B0_plugin_scale

echo "==== 运行 ===="
LD_LIBRARY_PATH=$MNN/build:$MNN/build/express ./B0_plugin_scale 2>&1 | \
    grep -vE "CPU Group|device supports"