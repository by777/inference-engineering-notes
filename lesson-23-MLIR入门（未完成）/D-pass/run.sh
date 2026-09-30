#!/bin/bash
# ============================================================
# run.sh：D-pass 常用命令
# 想看哪个效果，就取消那一行的注释（删掉行首的 #）再运行
# ============================================================

# 编译（改了 C++ 代码后运行）
# cmake -B build -G Ninja .
# ninja -C build

# 标准折叠：跑我们的 pass（推荐）
# ./build/dpass-opt --pass-pipeline="builtin.module(func.func(dpass-fold-addi))" test.mlir

# 只解析不优化：不指定 pass，原样打印 IR
# ./build/dpass-opt test.mlir

# 看帮助
# ./build/dpass-opt --help

# 从键盘输入 IR（敲完按 Ctrl-D 结束）
# ./build/dpass-opt --pass-pipeline="builtin.module(func.func(dpass-fold-addi))"
