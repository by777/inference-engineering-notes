# Lesson 22: MNN 算子实现与量化部署 总结

## 课程目标

1. 算子骨架：MNN 内置算子的三段式（InferShape / Execution / 注册宏）——实验A
2. 自定义算子：不改源码，用插件机制注册自己的算子——实验B
3. 量化推理：PTQ 量化 + 运行时自动 int8 通路——实验C
4. 白盒技能：不改框架源码，用官方回调 API 观察运行时行为

**呼应 Lesson 21**：量化公式（scale/zero_point/requantize）在 MNN 里怎么落地、运行时怎么自动执行 int8——从"原理"到"框架实现"的打通。

**呼应 Lesson 13-18（TVM）**：TVM 用调度器生成算子，MNN 用注册表挂载算子——不同框架的算子组织方式对比。

---

## 一、MNN 算子三段式（实验A 调用链 + 实验B 手写）

### 1. 内置算子骨架（实验A 用回调验证）

```
InferShape (shape 阶段: 算输出 shape, 分配内存)
   ↓
Execution 子类 (backend 阶段: onResize 准备 + onExecute 计算)
   ↓
Creator 工厂 (onCreate: 从 op 参数创建 Execution)
   ↓
REGISTER_CPU_OP_CREATOR(Creator, OpType_XXX)  (注册宏: 类型↔工厂绑定)
```

实验A 用回调 API 实测 depthwise conv 调用链：
```
输入 → Raster (NCHW→NC4HW4) → ConvolutionDepthwise → Raster (NC4HW4→NCHW) → 输出
```

**关键认知**：
- 回调 API `runSessionWithCallBackInfo(session, before, after)` 是"不改源码观察框架"的正路（libMNN.so 是 stripped 黑盒，改不了）
- `before` 回调拿到**输入** tensor，`after` 回调拿到**输出** tensor（Pipeline.cpp: `before(cmd.workInputs)`, `after(cmd.workOutputs)`）
- MNN 内部用 NC4HW4（通道按 4 打包）布局计算，转换时自动插 Raster

### 2. 插件算子三段式（实验B 手写 PluginScale）

```
① InferShapeKernel    (shape 阶段: 定输出 shape)
② CPUComputeKernel    (backend 阶段: init 读参数 + resize 准备 + compute 计算)
③ REGISTER_PLUGIN_OP / REGISTER_PLUGIN_COMPUTE_KERNEL  (注册宏)
```

PluginScale：`y = x*scale + bias`，scale/bias 用 attr 传。主程序用 Express API 直接构造 `OpT(OpType_Plugin) → PluginT(type="PluginScale", attr=[scale,bias])` 节点。

**关键认知**：
- 插件算子 = 内置算子同一套骨架，只是注册表不同（`ComputeKernelRegistry` / `InferShapeKernelRegister`）
- `PluginT.type` 字符串 ↔ 注册宏的字符串必须一致，运行时按名字查实现
- 注册宏展开是匿名命名空间的 static 对象，程序启动自动构造 → 自动注册，无需手动调用
- **不能从 ONNX 转换得到 plugin 算子**：转换器对不认识的节点（含 Custom）统一转成 `OpType_Extra`，不落 `OpType_Plugin`——必须用 Express API 直接构造（官方 test 也这么干）

---

## 二、MNN 的 int8 量化推理（实验C）

### 1. PTQ 量化流程（quantized.out）

```
float 模型 + 校准数据(图片)
   ↓ quantized.out
   ├─ weight_quantize_method=MAX_ABS: scale_w = max|w|/127  (不用数据)
   ├─ feature_quantize_method=KL: 跑模型收集激活分布 → 选最优截断阈值 → scale_act
   ↓
int8 模型 (量化参数写进 Conv 的 quanParameter)
```

校准图本质 = 模型的输入 tensor（quantized.out 只支持 image/sequence 输入，借图片通道）。`preprocessConfig.json` 里 `width/height` 必须匹配模型输入，`mean/normal` 是"像素→输入张量"的预处理（用户提供，不是 BN 层，MNN 不猜）。

### 2. int8 模型结构（dwconv_int8.json）

```
extraTensorDescribe: quantInfo { scale, zero, type: DT_INT8 }  每个 tensor 的量化信息
Conv 的 quanParameter:
  buffer  : int8 量化后的权重
  alpha   : per-channel 权重 scale (每输出通道一个)
  scaleIn : 输入 scale (KL 校准 ≈ 1/127)
  scaleOut: 输出 scale (requantize 用)
symmetricQuan: { nbits:8, zeroPoint:0, clampMin:-127, clampMax:127, outputDataType:DT_INT8 }
```

**关键**：权重已量成 int8，量化参数直接挂 Conv 上，**没有显式 FloatToInt8/Int8ToFloat 节点**——运行时就地 requantize。

### 3. 运行时自动 int8 通路（C3 回调实测）

```
JSON 静态结构:  ConvertTensor → ConvolutionDepthwise → ConvertTensor  (无 FloatToInt8!)
运行时回调实测: FloatToInt8[DT_INT8] → Raster → ConvolutionDepthwise[DT_INT8] → Raster
```

**关键**：调度器根据 tensor 的 quantAttr **运行时自动插入**量化转换算子——用户无感、模型侧不写死、调度时动态决定。算子名带 `[ DT_INT8 ]` 后缀 = 输出是 int8 类型。

完整链：`float 输入 → FloatToInt8 → ConvInt8(requantize) → Int8ToFloat → float 输出`。

### 4. 数值验证（C2）

```
float vs int8 最大绝对误差 = 0.0289 (ORT 参考 0.0156, 同量级)
```

**int8 模型输出必须 copyToHostTensor**：直接读 host<float>() 读到打包字节（垃圾），copyToHostTensor 自动做 NC4HW4→NCHW + int8→float 反量化。（官方 demo pictureRecognition.cpp 第 117 行的写法。）

### 5. 和 QDQ 假量化的区别

| 路线 | 结构 | 本质 |
|---|---|---|
| ONNX QDQ → MNN | `FloatToInt8 → Int8ToFloat → Conv(float)` | 假量化，Conv 还是 float |
| quantized.out → MNN | `Conv(带 quanParameter, 就地 int8)` | 真 int8 |

（ONNX QDQ 转 MNN 还有 scale 偏移 bug：内部按 uint8 域 +128 处理，转换器没抵消。**教训：MNN 真 int8 走 quantized.out**。）

---

## 三、白盒调试技能（贯穿三个实验）

1. **回调 API 是观察框架的正路**：libMNN.so stripped 改不了，`runSessionWithCallBackInfo` 挂钩子看每层执行（实验A 看调用链、实验C 看 int8 通路）
2. **MNNDump2Json 解剖模型结构**：看算子类型、量化参数（scaleIn/scaleOut/alpha）——Netron 看不到量化参数
3. **copyToHostTensor 读输出**：MNN 内部 NC4HW4 + 可能 int8 打包，直接读 host 是坑
4. **编译三件套 include 路径**：`include/` + `schema/current/`（生成头）+ `3rd_party/flatbuffers/include`；Express 符号要链 `libMNN_Express.so`（实验B 踩的坑）

---

## 四、和 Lesson 21 的打通

| Lesson 21 概念 | MNN 里的落地（实验C） |
|---|---|
| 对称量化 `q=round(x/scale)` | zero=0, scale=0.007872 |
| per-channel | 权重每输出通道一个 scale（alpha 数组） |
| requantize | int32 累加 × (scaleIn×alpha/scaleOut) → clamp → int8 |
| 分布天花板（KL） | 统计激活分布选最优截断点定 scale |
| PTQ / QAT | quantized.out 是 PTQ；quanDemo.py 是 QAT |

## 五、和 Lesson 13-18（TVM）的打通

| | TVM | MNN |
|---|---|---|
| 算子怎么来 | TE 调度 + 代码生成 | 注册表 + Execution 实现 |
| 算子怎么组织 | `compute/schedule` 声明式 | `InferShape + Execution` 三段式 |
| 自定义算子 | `te.extern` / 手动 codegen | `REGISTER_PLUGIN_*` 插件 |
| 优化粒度 | 调度原语（vectorize/unroll/并行） | 后端实现内嵌（NEON/AVX）+ 几何管线 |

**共性**：都是"算子注册表 + 执行器"模式——框架核心不写死每个算子，靠注册机制挂载，这就是 AI 框架可扩展性的根基。

---

## 六、踩坑汇总（三个实验的教训）

1. **sh vs bash**：`source` 在 sh(dash) 里不可用 → 用 bash
2. **缺头文件**：`<chrono>`、`Tensor_generated.h`（在 schema/current/）、`flatbuffers.h`（在 3rd_party）→ 编译报错逐个补 include 路径
3. **Express 符号未定义**：`_Input`/`Expr::create` 在 `libMNN_Express.so`（build/express/ 子目录），默认不链 libMNN → 加 `-lMNN_Express`
4. **回调 before 读输出是坑**：before 拿到输入 tensor，读输出要 after（实验C 踩过）
5. **y_scale 不能拍脑袋估算**：conv 输出幅度 = 输入×权重×累加项数，必须校准统计（实验C 坑 2）
6. **int8 输出不能直接读 host**：必须 copyToHostTensor（实验C 坑 3，最隐蔽）
7. **QDQ 不是 MNN int8 主路**：假量化 + scale 偏移 bug，真 int8 走 quantized.out（实验C 坑 1）

## 七、三个实验的衔接关系

```
实验A: 观察浮点调用链 (回调 API + Raster/Conv)
   ↓  "算子怎么被调度执行"
实验B: 手写插件算子 (三段式注册, 自己造算子进图)
   ↓  "算子怎么注册/实现"
实验C: int8 量化推理 (PTQ 量化 + 自动 int8 通路)
   ↓  "算子怎么量化部署"
```
