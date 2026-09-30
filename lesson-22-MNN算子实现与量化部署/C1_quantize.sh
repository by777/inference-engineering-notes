#!/bin/bash
# C1_quantize.sh
# 实验C: 用 MNN 官方量化工具 quantized.out 把 float 模型量化为 int8 模型
#
# 为什么需要这一步?
#   MNN 的真 int8 推理, 模型必须用官方量化工具(或训练量化)生成 ——
#   量化参数(scaleIn/scaleOut/alpha)会写进模型, 运行时就地 requantize。
#   (ONNX QDQ 转 MNN 得到的是"假量化"链, Conv 还是 float, 不是真 int8!)
#
# 校准: quantized.out 需要校准数据统计激活分布定 scale,
#       这里用 C0_make_calib.py 生成的 calib/ 里的 8x8 图。
#       配置见 preprocessConfig.json (width/height=8, KL 校准)。
#
# 产物: dwconv_int8.mnn (int8 模型) + dwconv_int8.json (结构反解, 解剖用)
#
# 用法: bash C1_quantize.sh

set -e
cd "$(dirname "$0")"

echo "==== ① 量化: float 模型 → int8 模型 ===="
# quantized.out 的三个参数:
#   参数1 (dwconv_float.mnn)  : 原始 float 模型 (待量化的模型)
#   参数2 (dwconv_int8.mnn)   : 输出的 int8 模型 (量化产物)
#   参数3 (preprocessConfig.json): 校准配置 (校准图目录 + 量化方法 + 输入尺寸)
../mnn-src/build/quantized.out \
    dwconv_float.mnn \
    dwconv_int8.mnn \
    preprocessConfig.json

echo ""
echo "==== ② 反解 int8 模型结构 (解剖用) ===="
# MNNDump2Json 的两个参数:
#   参数1 (dwconv_int8.mnn) : 要反解的 int8 模型
#   参数2 (dwconv_int8.json): 输出的 JSON 结构 (看算子类型/量化参数用)
../mnn-src/build/MNNDump2Json dwconv_int8.mnn dwconv_int8.json

echo ""
echo "==== ③ 确认 int8 结构: 量化参数 (scaleIn/scaleOut/alpha) ===="
# 从 JSON 里抓量化参数, 确认量化信息写进了模型
#   scaleIn  : 输入 scale (KL 校准得到, 约 1/127)
#   scaleOut : 输出 scale (requantize 用)
#   alpha    : per-channel 权重 scale (每输出通道一个)
grep -E "scaleIn|scaleOut|alpha|tensorScale|symmetricQuan" dwconv_int8.json | head -10

echo ""
echo "完成! 产物: dwconv_int8.mnn + dwconv_int8.json"
