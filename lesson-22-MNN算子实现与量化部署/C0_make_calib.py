# C0_make_calib.py
# Lesson 22 实验C: 生成 8x8 RGB 校准图 (给 quantized.out 做量化校准)
#
# 为什么需要校准图?
#   MNN 的量化工具 quantized.out 需要"校准数据"来统计每个中间 tensor 的
#   数值分布, 从而确定量化 scale (这就是 Lesson 21 讲的"定标/校准")。
#   我们不搞真实数据集 —— 造一张能反映输入范围的 8x8 小图就够了。
#
# 校准图怎么造?
#   模型输入是 [1,3,8,8] (NCHW), 所以造一张 8x8 的 RGB 图。
#   像素值分布和后面 C2 推理用的输入同分布 (都是 (i%17)*0.01-0.08 映射到 0~255),
#   这样校准出来的 scale 才能反映真实输入的分布。
#
# 用 BMP 格式: 24-bit 无压缩, 格式简单可以手写字节, 不需要 PIL 库。

import numpy as np
import struct
import os

os.makedirs("calib", exist_ok=True)  # 校准图目录

w = h = 8
data = np.zeros((h, w, 3), dtype=np.uint8)  # 8x8x3 的 RGB 数组

# 填像素: 让像素值和 C2 的推理输入同分布
# C2 输入公式: (i%17)*0.01 - 0.08, 范围 [-0.08, 0.08]
# 映射到 [0,255]: (v + 0.08) / 0.16 * 255
for i in range(h * w):
    v = (i % 17) * 0.01 - 0.08  # 原始值, 范围 [-0.08, 0.08]
    p = int((v + 0.08) / 0.16 * 255)  # 映射到 [0,255]
    r, c = divmod(i, w)  # 行列
    data[r, c] = [p, p, p]  # 灰度图 (RGB 三个通道相同)

# 写 BMP 文件 (24-bit 无压缩)
with open("calib/calib_0.bmp", "wb") as f:
    row_size = (w * 3 + 3) & ~3  # 每行字节数要 4 字节对齐
    pixel_size = row_size * h  # 像素数据总大小
    file_size = 54 + pixel_size  # 文件总大小 = 头部(54) + 像素

    f.write(b"BM")  # BMP 魔数
    f.write(struct.pack("<I", file_size))  # 文件大小
    f.write(struct.pack("<HH", 0, 0))  # 保留字段
    f.write(struct.pack("<I", 54))  # 像素数据偏移 (头部 54 字节)

    # 信息头 (40 字节)
    f.write(struct.pack("<I", 40))  # 信息头大小
    f.write(struct.pack("<ii", w, h))  # 宽高 (有符号)
    f.write(struct.pack("<HH", 1, 24))  # 颜色平面数=1, 每像素位数=24
    f.write(struct.pack("<I", 0))  # 压缩方式=0 (无压缩)
    f.write(struct.pack("<I", pixel_size))  # 像素数据大小
    f.write(struct.pack("<ii", 2835, 2835))  # 分辨率
    f.write(struct.pack("<II", 0, 0))  # 调色板信息

    # 像素数据: BMP 存储是"从下往上"的, 所以要倒着写行
    for r in range(h):
        row = np.zeros(row_size, dtype=np.uint8)  # 一行 (含填充)
        row[: w * 3] = data[h - 1 - r].reshape(-1)  # 倒序取行
        f.write(row.tobytes())

print("生成 calib/calib_0.bmp (8x8 RGB)")
