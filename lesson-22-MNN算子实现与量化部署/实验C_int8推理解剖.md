# Lesson 22 实验C: MNN int8 量化推理解剖

## 文件清单（全部跑通）

| 文件 | 作用 | 怎么验证的 |
|---|---|---|
| `C0_make_calib.py` | 生成 8x8 RGB 校准图 | 跑完生成 `calib/calib_0.bmp` |
| `preprocessConfig.json` | 量化工具的校准配置 | `C1_quantize.sh` 读它成功 |
| `C1_quantize.sh` | quantized.out 量化 float→int8 + 反解 JSON | 输出 `Quantize model done!` |
| `C2_run.cpp` / `C2_build.sh` | float vs int8 输出数值对比 | max diff = 0.0289 |
| `C3_trace.cpp` / `C3_build.sh` | 回调观察 int8 运行时执行链 | 见第 4 节 |
| `dwconv_float.mnn` / `dwconv_int8.mnn` / `.json` | 产物（float 基准 / int8 模型 / 结构反解） | — |

## 目标

用 MNN 官方量化工具 `quantized.out`（PTQ 训练后量化）把 float 模型量化为 int8 模型，解剖：
1. 量化参数（scaleIn/scaleOut/alpha）怎么算出来、写进模型哪里
2. 运行时真正执行哪些算子（MNN 怎么自动调度 int8 通路）
3. 量化引入多大误差（float vs int8 输出对比）

## 一、校准图：为什么需要、是什么

### 为什么需要校准数据？

量化有两个 scale 要定：
- **权重 scale**：直接看权重值 `max|w|/127`（MAX_ABS），不需要数据
- **激活 scale**：取决于输入长什么样，**模型静态推不出来** → 必须喂数据跑一遍，统计中间 tensor 分布（KL 算法选最优截断点）

### 校准图本质就是模型的输入 tensor

```
calib_0.bmp (8x8 RGB, uint8 像素 [0,230])
   │  ImageProcess: dst = (src - mean) × normal     ← mean/normal 在这里用
   ▼
输入 tensor [1,3,8,8] (float, 值 [0,0.9])
   │  喂给 float 模型
   ▼
跑一遍, 收集中间分布 → KL 定 scale → 写进模型
```

- 校准图 = "图片化的输入 tensor"，quantized.out 只支持 image/sequence 输入，所以借图片通道
- 我们造的图像素分布和 C2 推理输入同分布（`(i%17)*0.01-0.08` 映射到 [0,255]），scale 才准
- **脚本里看不到校准图**：它不在命令行参数里，而是 `preprocessConfig.json` 的 `path: "calib"` 字段引用的（工具内部 `readClibrationFiles` / `preprocessInput` 去读）

### mean/normal 是什么

- **mean/normal 是"像素→输入张量"的数据预处理**，不是量化参数，也不是模型里的 BN 层
- 它描述"图片怎么变成模型输入"，这个知识属于**用户**（来自训练/部署约定），MNN 不猜、不自己算
- 真实视觉模型训练时有 `Normalize(mean, std)`（PyTorch 数据增强，模型 forward 之前）；BN 是模型内的层（参数在模型里）
- 我们的 dwconv **没有**预处理 → mean=0、normal=1（恒等映射）；但我们 normal=1/255，因为 BMP 像素是 uint8 [0,255]，要除 255 还原成 float [0,1]——这是"借道图片格式"逼的，不是做了 Normalize

## 二、量化命令与配置（C1_quantize.sh）

```bash
../mnn-src/build/quantized.out dwconv_float.mnn dwconv_int8.mnn preprocessConfig.json
```
三个参数：①待量化 float 模型 ②输出的 int8 模型 ③校准配置（校准图目录 + 量化方法 + 输入尺寸）。

`preprocessConfig.json` 关键字段：
| 字段 | 值 | 含义 |
|---|---|---|
| `width/height` | 8/8 | **必须匹配模型输入 H/W** |
| `path` / `used_image_num` | calib / 1 | 校准图目录 / 用几张 |
| `feature_quantize_method` | KL | 激活量化：统计分布选最优截断阈值 |
| `weight_quantize_method` | MAX_ABS | 权重量化：`max\|w\|/127` 对称量化 |
| `mean/normal` | 0 / 1/255 | 像素→输入张量的预处理（见上） |

编译前提：quantized.out 需 `-DMNN_BUILD_QUANTOOLS=ON` 重编（默认 OFF）。

## 三、int8 模型结构（dwconv_int8.json）

### 1. extraTensorDescribe：每个 tensor 的量化信息

```
index 0: quantInfo { scale: 0.007872, zero: 0, type: DT_INT8 }   ← 输入
index 2: quantInfo { scale: 0.045033, ... }                       ← Conv 输出
```

**scale 是校准算出来的**：输入 0.007872 ≈ 1/127，输出 0.045033（KL 从校准图的激活分布选出）。

### 2. Conv 的 quanParameter：就地 int8 计算 + requantize

```
"quanParameter": {
    "buffer": [ 2, 3, 0, 9, ... -28, -105, -73, ... ],  // int8 量化后的权重
    "alpha": [ 0.017645, 0.011764, 0.020102 ],          // per-channel 权重 scale
    "scaleIn": 0.007872,                                 // 输入 scale
    "scaleOut": 0.045033,                                // 输出 scale (requantize 用)
}
"symmetricQuan": { nbits: 8, zeroPoint: 0, clampMin: -127, clampMax: 127, outputDataType: DT_INT8 }
```

**关键**：权重已量成 int8，量化参数直接挂 Conv 上，**没有显式 FloatToInt8/Int8ToFloat 节点**。运行时就地 requantize（int32 累加 × scaleIn×alpha/scaleOut → clamp → int8）。

### 3. 和 QDQ 假量化的区别（踩坑结论）

| 路线 | 结构 | 本质 |
|---|---|---|
| ONNX QDQ → MNN | `FloatToInt8 → Int8ToFloat → Conv(float)` | 假量化，Conv 还是 float |
| **quantized.out → MNN** | `Conv(带 quanParameter, 就地 int8)` | 真 int8 |

（ONNX QDQ 转 MNN 还会踩 scale 偏移 bug：内部按 uint8 域 +128 处理，转换器没抵消 → 数值错。**教训：MNN 真 int8 走 quantized.out，不走 ONNX QDQ**。）

## 四、运行时执行链（C3_trace，回调实测）

```
[before] FloatToInt8 [ DT_INT8 ]   name=output_raster_0     ← 运行时自动插入!
[before] Raster [ DT_INT8 ]                                 ← NC4HW4 布局转换
[before] ConvolutionDepthwise [ DT_INT8 ]   name=output      ← int8 卷积 + 就地 requantize
[before] Raster [ DT_INT8 ]                                 ← int8→float 反量化 + 布局转回
```

**核心结论**：`dwconv_int8.json` 里没有 FloatToInt8 节点，但运行时回调看到了——**MNN 调度器根据 tensor 的 quantAttr 自动插入量化转换算子**。这就是"自动 int8 通路"：用户无感、模型侧不写死、调度时动态决定。算子名带 `[ DT_INT8 ]` 后缀 = 输出是 int8 类型。

完整链：`float 输入 → FloatToInt8 → ConvInt8(requantize) → Int8ToFloat → float 输出`，对用户透明。

## 五、float vs int8 数值对比（C2_run）

```
=== float vs int8 输出对比 (前 12 个值) ===
  [0] float=-0.0820274  int8=-0.0900654  diff=0.00803798
  [1] float=-0.254942  int8=-0.270196  diff=0.0152543
  ...
最大绝对误差 = 0.0289145 (下标 13)
参考: ORT 标准 QDQ 误差 ≈ 0.0156
```

int8 数值正确（0.029 和 0.0156 同量级），误差来自输入/输出/权重量化舍入 + KL 校准的统计近似。

### ⚠️ 最大的坑：int8 模型输出不能直接读 host 指针

```
直接 host<float>()   → int8 字节被当 float 解释 (垃圾值)
直接 host<int8_t>()  → 打包后的 int8 值, 没反量化, 格式也不对
正确: copyToHostTensor → 自动 NC4HW4→NCHW + int8→float 反量化
```

```cpp
auto dimType = outputTensor->getDimensionType();
if (outputTensor->getType().code != halide_type_float) {
    dimType = Tensor::TENSORFLOW;
}
std::shared_ptr<Tensor> outHost(new Tensor(outputTensor, dimType));
outputTensor->copyToHostTensor(outHost.get());  // ★ 关键
auto outPtr = outHost->host<float>();
```

（参考官方 demo/exec/pictureRecognition.cpp 第 117 行。`getType()` 可能仍标 float，但 host 内存是 int8 打包字节——所以必须 copy。）

## 六、从实验结果能确认什么（都是亲手验证的）

1. **MNN 真 int8 = 就地 requantize**：量化参数挂 Conv 上，运行时 int8 计算 + 就地 requantize，不经过中间 float——和 ONNX QDQ"假量化"本质区别
2. **PTQ 自动量化**：float 模型 + 校准数据 → quantized.out（MAX_ABS 定权重 scale + KL 定激活 scale）→ int8 模型，运行时 MNN 自动调度 int8 通路，**用户无感、输出仍 float**
3. **scale 是校准出来的**：输入 0.007872、输出 0.045033、权重 per-channel alpha——衔接 Lesson 21 的"量化三要素"
4. **运行时自动插量化转换**：JSON 没有 FloatToInt8，回调看到——调度器按 quantAttr 动态决定
5. **读输出必须 copyToHostTensor**：MNN 内部 NC4HW4 + int8 打包，直接读 host 是坑
6. **校准图 = 输入 tensor 的图片化载体**：mean/normal 是用户提供的数据预处理（非 BN 层），scale 才是工具算的量化参数

## 七、衔接 Lesson 21

Lesson 21 的量化公式在 MNN 里全部落地：
- **对称量化** `q = round(x/scale)`：zero=0，scale=0.007872
- **per-channel**：权重每输出通道一个 scale（alpha 数组）
- **requantize**：int32 累加 × (scaleIn×alpha/scaleOut) → clamp → int8（Lesson 21 的 multiplier 思路）
- **KL 校准**：统计激活分布定 scale（Lesson 21 的"分布天花板"，不是用 max，而是选 KL 散度最小的截断点）
- **PTQ vs QAT**：quantized.out 是 PTQ（不需要训练）；quanDemo.py 是 QAT（训练时模拟量化误差，精度更高）

## 踩坑路线：从 ONNX QDQ 死磕到 MNN 官方主路

实验 C 走了弯路：最初选 ONNX QDQ 路线，死磕 MNN 的 QDQ 兼容层（发现假量化链 + scale 偏移 bug）后及时转向 MNN 官方主路（quantized.out），才解剖到真正的 int8 推理。中途踩了 3 个坑：
1. QDQ 不是 int8 主路（假量化 + 偏移 bug）→ 换 quantized.out
2. y_scale 拍脑袋估算导致饱和 → 必须校准（校准图统计真实分布）
3. int8 输出直接读 host 读到垃圾 → 必须 copyToHostTensor

最终所有结论都是亲手验证的：C2 数值（0.029）、C3 执行链（FloatToInt8[DT_INT8] → Conv[DT_INT8]）、JSON 结构（scaleIn/scaleOut/alpha）。
