# Lesson 22 实验A: MNN depthwise conv 调用链

## 文件清单（全部跑通）

| 文件 | 作用 | 怎么验证的 |
|---|---|---|
| `A0_gen_model.py` | 生成 depthwise conv 的 onnx | 跑完打印"生成 dwconv.onnx" |
| `A1_convert_check.sh` | ONNX→MNN + 确认算子类型 | 输出见下 |
| `A2_build.sh` | 编译 + 运行 C++ 推理 | 编译零报错 + 推理成功 |
| `A3_run.cpp` | MNN C++ 推理程序 | 0.012ms，输出 16 个值 |
| `A4_trace.cpp` | 逐层调用链跟踪（回调钩子） | 见第 4 节 |

## 我实际跑出来的结果

### 1. 生成了什么模型

```
输入 [1,3,8,8] → Conv(group=3, kernel 3x3, pads=1) → 输出 [1,3,8,8]
权重 [3,1,3,3]
```

`A0_gen_model.py` 里的关键：
- 输入 `[1, 3, 8, 8]` = N C H W
- 权重 `[3, 1, 3, 3]` = OC IC KH KW（IC=1 → depthwise）
- `group=3` 使 `C_in/group = 3/3 = 1`，权重 IC 维 = 1 → depthwise

### 2. 转换后确认算子类型（A1_convert_check.sh 输出）

```
2 "type": "ConvertTensor"         ← 布局转换 (NCHW → NC4HW4)
1 "type": "ConvolutionDepthwise"  ← depthwise conv ✓
1 "type": "Input"
```

**这说明**：ONNX 的 `Conv(group=3)` 被 MNN 转换器识别成了 `ConvolutionDepthwise`，
写进了 .mnn 文件。

### 3. 推理跑通（A2_build.sh 输出）

```
[0] 模型加载成功
[1] 会话创建成功 (CPU 后端)
[2] 输入 tensor shape=1,3,8,8,
[3] 推理完成, 耗时 0.012196 ms
[4] 输出前 16 个值: -0.0820274 -0.254942 -0.216675 -0.178408 -0.140141 -0.101874
 -0.0636075 -0.0170831 -0.114148 -0.133388 -0.225207 -0.155511 -0.0858145 -0.0161182
 0.0535782 0.200425
```

### 4. 调用链验证（A4_trace.cpp，回调钩子实测）

**目标**：亲眼看 MNN 推理时算子按什么顺序执行——运行时证据，不是读源码猜的。

**方法**：用 MNN 官方回调 API `runSessionWithCallBackInfo(session, before, after)`。
它在每层执行前调 `before`、执行后调 `after`，回调里能拿到算子的 `type` 和 `name`。
**零源码改动**（libMNN.so 是 stripped 黑盒，改不了；官方钩子才是正路）。

**实测输出**：

```
[1] 开始推理 (逐层跟踪, 含算子类型):
  [before] type=Raster  name=output_raster_0  in_shape=1,3,8,8,
  [after]  type=Raster  name=output_raster_0
  [before] type=ConvolutionDepthwise  name=output  in_shape=1,3,8,8,
  [after]  type=ConvolutionDepthwise  name=output
  [before] type=Raster  name=output__before_tr_raster_0  in_shape=1,3,8,8,
  [after]  type=Raster  name=output__before_tr_raster_0
```

**从输出读出的调用链**（按执行顺序）：

```
输入
  → Raster (NCHW → NC4HW4)        输入端布局转换
  → ConvolutionDepthwise           真正的 depthwise conv ✓
  → Raster (NC4HW4 → NCHW)        输出端布局转换
输出
```

**关键认知**：
- 转换期 JSON 里看到的 `ConvolutionDepthwise`（静态结构），运行期真的被执行了
  （动态证据）——"识别→调度→执行"整条链路成立
- Raster 是 MNN 的布局转换算子（NCHW↔NC4HW4），被自动插在 conv 前后
- 回调返回 `true` = 继续执行该层；返回 `false` = 跳过该层（钩子不只观察还能干预）

**这个技能的价值**：不改框架源码就能观察框架内部行为——在公司改不了 libMNN.so
时，这就是调试算子的正路。

## 从这几个结果能确认什么（都是亲眼见的）

1. **ONNX 怎么表示 depthwise**：`group == C_in`（都是 3），权重 IC 维 = 1。
   每个输出通道只卷自己那一个输入通道，通道间不混合。
   （衔接 Lesson 21：fsmn 的 conv_left 权重 `(128,1,10,1)` 就是 depthwise）

2. **转换期识别了算子类型**：ONNX `Conv(group=3)` → MNN `ConvolutionDepthwise`。
   这个类型写进了 .mnn 文件，运行时靠它找实现。

3. **MNN 转换时插入了布局转换**：`ConvertTensor`（NCHW→NC4HW4）出现在 conv
   前后。说明 MNN 内部用 NC4HW4（通道按 4 打包）布局计算——这是 MNN 的一个
   重要特征，具体原因实验B 接触源码时再深挖。