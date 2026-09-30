#!/bin/bash
# run_bench.sh —— Lesson 11 测试命令备忘
# 用法: ./run_bench.sh [次数] [线程数]

ITERS="${1:-200}"    # 第1个参数: 测试次数, 默认 200
THREADS="${2:-4}"    # 第2个参数: intra线程数, 默认 4

# 0. 准备
python3 gen_model.py      # 生成模型 (已存在会覆盖)
make main                 # 编译

# 1. bench 测速
echo "====== bench: $ITERS 次, intra_threads=$THREADS ======"
./main bench "$ITERS" "$THREADS"

# 2. profile 导出热点 (跑 20 次, 固定 4 线程)
echo "====== profile: 导出热点 ======"
rm -f profile.json_*.json
./main profile 20
PROFILE_FILE=$(ls profile.json_*.json | head -1)
python3 analyze_profile.py "$PROFILE_FILE"
