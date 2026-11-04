# Lesson 26：onnxsim —— 离线图化简

> ORT 的图优化会产出 `com.microsoft.nchwc:*` 这类私有算子，绑死环境。
> 这一课走另一条路：**离线、产物仍是标准 ONNX、可以发给别人**的图化简。
>
> 分工一句话：**onnxsim 让图变小且标准，ORT 让图贴合这台机器。**

---

## 一、为什么要单独学它

"图优化"是个被用滥的词。实际至少分四层，每层的时机、产物、可移植性都不同：

| 层 | 时机 | 产物算子域 | 能否分发 |
|---|---|---|---|
| ① 导出期 `torch.onnx.export` | 建模时 | `ai.onnx`（常带 `Identity`/`Cast` 垃圾） | 能 |
| ② **离线化简（onnxsim）** | **部署前跑一次** | **`ai.onnx` 标准算子** | **能** |
| ③ 加载期图变换（ORT 三级） | 每次建 Session | 可能 `com.microsoft` / `nchwc` | 视级别而定 |
| ④ EP 分割 / kernel 选择 / 预打包 | 每次建 Session | — | 不能 |

只有第 ② 层既优化了、又能分发。这就是 onnxsim 的位置。

和仓库其它课的关系：

| 课 | 关系 |
|---|---|
| Lesson 07（SessionOptions 与图优化控制） | 那是第 ③ 层；本课是第 ② 层 |
| Lesson 09 / 10（Session 内省 / 保存加载） | 那两课的工具可以观察本课产物 |
| Lesson 13–18（TVM）、Lesson 22（MNN） | onnxsim 产物是标准 ONNX，直接能喂 |

---

## 二、它做什么

默认 40 个 pass（`python3 -m onnxsim --list-default-optimizers`）：

| 类别 | 数量 | 代表 |
|---|---|---|
| `eliminate_*` 消除 | 21 | `eliminate_identity`、`eliminate_nop_cast`、`eliminate_common_subexpression`、`eliminate_deadend` |
| `fuse_*` 融合 | 16 | `fuse_bn_into_conv`、`fuse_matmul_add_bias_into_gemm`、`fuse_pad_into_conv`、`fuse_qkv` |
| 其它 | 3 | `nop`、`extract_constant_to_initializer`、`rewrite_where` |

实验 1 的实测（8 节点 → 3 节点）：

```
原始   Identity, Cast, MatMul, Add, Mul, Mul, Div, Identity   4 个 initializer
简化   Gemm, Mul, Div                                         3 个 initializer
k = 6.0（Mul(2.0, 3.0) 被折叠）
```

三类动作同时发生：删 `Identity`/空 `Cast`、折叠常量、融合 `MatMul+Add → Gemm`。

注意 `Mul(x,k)` / `Div(·,k)` **没有被约掉**。浮点下 `(x*k)/k ≠ x`，onnxsim 不做这种不安全化简。

**"没被优化"有时是工具做对了**，跟"漏优化"要分开看。

---

## 三、迭代到不动点

这是 onnxsim 和"跑一遍 pass"的本质区别。

### 3.1 为什么需要迭代

看源码 `onnxsim/onnxsim.cpp` 的组装（v0.7.3，0.8.x 结构相同）：

```cpp
// 第一层不动点：shape 推理与 optimizer 交替
ModelFn OptAndShape = FixedPointFn(InferShapes, OptimizeInPlace, fixed_point_iters);

// 第二层不动点：上面这个整体 与 常量折叠 交替
ModelFn Pipeline = FixedPointFn(OptAndShape, FoldConstant, fixed_point_iters);
```

`FixedPointFn` 的语义是「交替跑 f1、f2，直到模型指纹不再变化」。上限默认 50，可用 `ONNXSIM_FIXED_POINT_ITERS` 改。

### 3.2 一遍不够的原因：融合的前提是另一个 pass 的产出

`e2.py` 造的图：3 层 BN 串行，每层 scale 由上一层的 `Mul` 算出。

```
Conv → Mul(a,b)=sc1 → BN(sc1) → Mul(sc1,c2)=sc2 → BN(sc2) → Mul(sc2,c3)=sc3 → BN(sc3)
```

`fuse_bn_into_conv` 需要 BN 的 scale 是**常量**才能算新权重：

$$
W' = W\cdot\frac{s}{\sqrt{\sigma+\epsilon}}
$$

但 `sc1` 是 `Mul` 的输出，是中间张量。融合 pass 看到它时只能说：「形状对上了，但 scale 不是常量，算不出新权重，跳过。」

随后常量折叠把 `a*b` 算成 `1.0`，`sc1` 才变成 initializer。**但这一轮的融合已经跑完了。**

折叠的产出只能被**下一轮**的融合看到。链条有三节，所以至少要三轮。

### 3.3 实测：每多一轮推进一层

onnxsim 0.8.1 的实测：

| iters | 节点 | inits | 已收敛 |
|---|---|---|---|
| 1 | `Conv, BN, BN, BN` | 6 | 否 |
| 2 | `Conv` | 2 | 否 |
| 3 | `Conv` | 2 | 是 |
| 10 | `Conv` | 2 | 是 |

手动循环调用（每次只给 1 轮）的结果与之对应：

```
第 1 次: ['Conv', 'BatchNormalization', 'BatchNormalization', 'BatchNormalization']
第 2 次: ['Conv']
第 3 次: ['Conv']
```

这证明：**「迭代到不动点」不是特殊算法，就是重复调用直到不变。** onnxsim 只是帮你做了这件事，并加了提前停止。

### 3.4 版本差异

同样的图，onnxsim **0.7.3** 需要 5 轮才报收敛，**0.8.1** 只要 3 轮就收敛，而且 `iters=1` 时 0.7.3 融掉一层、0.8.1 一层都没融。

内部不动点嵌套的层数不同，单轮内推进的步数也不同。所以：

- **不要硬编码"需要几轮"**，看警告就行
- **做对照实验必须固定版本**，否则结论不成立

### 3.5 警告的含义：结果正确 ≠ 已收敛

`iters=3` 时图已经是 `['Conv']`（正确），但可能仍在报：

```
WARNING: the simplification stopped because of timeout.
```

原因在 `FixedPointFn` 的判据：收敛定义为「**这次跑完和上次跑完一样**」，需要一轮确认。所以「结果正确」比「报告收敛」早到。

实践含义：**看到 WARNING 先确认结果对不对，再决定要不要调大上限。**

### 3.6 一个抓不到警告的坑

这个警告是 C++ 层**直接写 fd 1** 的，Python 的 `redirect_stdout` / `redirect_stderr` **都拦不住**：

```python
with contextlib.redirect_stdout(buf):
    r = onnxsim.simplify(m)     # 警告照样打到终端
print("WARNING" in buf.getvalue())   # False
```

要检测它只能用 `subprocess` 读子进程的原始输出。`e2.py` 就是这么做的。

---

## 四、四条边界

`e3.py` 逐条验证。判断标准统一：**改动会不会改变模型的语义或契约。**

### 4.1 动态 shape：默认保留

```
原始 : Shape, Gather, Concat, Reshape, Identity
化简 : Shape, Gather, Concat, Reshape
输入第 0 维 = 'batch'
  batch=1 / 3 / 8 都能跑
```

只有 `Identity` 被删（与 shape 无关）。**模型契约没变。**

### 4.2 `overwrite_input_shapes`：一用就焊死

同一个模型，传 `overwrite_input_shapes={"X": [1,4]}`：

```
化简 : Reshape
输入第 0 维 = 1
  batch=1 跑通
  batch=3 失败 -> INVALID_ARGUMENT: Got invalid dimensions
```

整条 `Shape` 链消失，因为 `[1,4]` 已知，全都能折成常量。

这不是"算错了"，是**图里写死了 1、你给 3，模型直接拒绝加载**。

| | 节点 | 第 0 维 | batch=3 |
|---|---|---|---|
| 默认 | `Shape, Gather, Concat, Reshape` | `'batch'` | 通过 |
| overwrite | `Reshape` | `1` | 失败 |

**这个参数是"性能 vs 通用性"的开关，默认关闭。** 固定 batch 的边缘部署可以加，通用服务不能加。

### 4.3 自定义算子：不穿透

```
原始   : ai.onnx:Identity, my_domain:MyAdd, ai.onnx:Mul, ai.onnx:Div, ai.onnx:Identity
化简后 : my_domain:MyAdd, ai.onnx:Mul, ai.onnx:Div
```

两头的 `Identity` 删了（纯结构优化，不需要语义），`MyAdd` 前后的 `Mul`/`Div` 一动不动（代数优化，需要语义）。

**为什么"不动"是对的**：onnxsim 不知道 `MyAdd` 是幂等、可交换、还是有副作用。万一它实现里会修改 `k`，约掉 `Mul`/`Div` 就错了。不认识的算子当一堵墙，是保守但正确的选择。

想让它认识，用 `--custom-lib libmy_ops.so`。但**自定义算子库的 ORT ABI 版本必须和当前运行时一致**，否则报 `The requested API version [N] is not available`。

### 4.4 量化 QDQ：完全不碰

```
原始   : QuantizeLinear, DequantizeLinear, Mul, Div
化简后 : QuantizeLinear, DequantizeLinear, Mul, Div
```

一个节点都没动。`scale` / `zero_point` 是量化语义的一部分，而且 **Q 是有损的**：

```
0.1234 --quantize(scale=0.05)--> int8 2 --> dequantize --> 0.1   精度已丢
```

约掉 QDQ 等于把量化模型变回浮点模型，数值完全不同。量化图的化简要用 ORT quantization tool / NNCF 一类专门工具。

### 4.5 小结

| 情况 | 行为 | 是否合理 |
|---|---|---|
| 动态维 | 保留动态 | 正确 |
| overwrite 静态 | 焊死 | 危险，但用户主动要求 |
| 自定义算子 | 当墙 | 正确 |
| QDQ | 不碰 | 正确 |
| `(x*k)/k` | 不约 | 正确（浮点不等价） |

---

## 五、和 ORT 的分工

`e4.py` 对比三条路。同一张 `Conv+BN+Relu`：

| 做法 | 产物 |
|---|---|
| 原始 | `ai.onnx:Conv, ai.onnx:BatchNormalization, ai.onnx:Relu` |
| 只跑 onnxsim | `ai.onnx:Conv, ai.onnx:Relu` |
| 只跑 ORT(ALL) | `com.microsoft.nchwc:Conv, com.microsoft.nchwc:ReorderOutput` |
| onnxsim → ORT | `com.microsoft.nchwc:Conv, com.microsoft.nchwc:ReorderOutput` |

两条路都完成了融合，但产物域不同。三种做法的数值都 `bit-equal=True`。

### 5.1 收益重叠，不是叠加

`MatMul+Add` 更能说明问题：

```
原始      MatMul, Add
onnxsim   Gemm
仅 ORT    Gemm
sim→ORT   Gemm
```

onnxsim 先合成了 `Gemm`，ORT 的 `MatMulAddFusion` 就找不到目标形状了。**叠加不会带来额外收益。**

### 5.2 融合的物理形态：语义变成属性

`com.microsoft.nchwc:Conv` 的真实内容：

```
inputs = ['X', 'reorder', 'ConvBnFusion_BN_B_bb']
attr activation    = b'Relu'
attr kernel_shape  = [3, 3]
attr pads          = [1, 1, 1, 1]
```

`Relu` 节点消失，变成 `Conv` 的 `activation` 属性。带参数的激活还会多一个字段：

```
attr activation        = b'LeakyRelu'
attr activation_params = [0.009999999776482582]
```

### 5.3 为什么 onnxsim 不产出私有算子

融合算子必须是**已注册的实体**，不能只是抽象符号。实测造一个不存在的：

```
com.microsoft:FusedNotExist  ->  FAIL: is not a registered function/op
```

`nchwc` 依赖 CPU EP 里对应的 kernel。换 GPU、换 ARM、换 ORT 版本，kernel 可能不存在，模型直接加载失败。

ORT 自己会警告：

> Serializing optimized model with Graph Optimization level greater than ORT_ENABLE_EXTENDED and the NchwcTransformer enabled. The generated model may contain hardware specific optimizations.

**onnxsim 保持标准算子不是能力不足，是维持"通用化简器"的定位。**

### 5.4 ORT 融合的约束（实测）

`ConvActivationFusion` 的激活白名单（从 `.so` 提取的字面量）：

```
Affine, Relu, LeakyRelu, ThresholdedRelu, Tanh, ScaledTanh,
Sigmoid, HardSigmoid, Elu, Softsign, Softplus
```

实测行为：

| 图 | ORT(ALL) |
|---|---|
| `Conv → Relu` / `Sigmoid` / `Tanh` / `HardSigmoid` / `LeakyRelu` / `Clip` | 融合 |
| `Conv → Softmax` | 不融合（不在名单） |
| `Conv → Elu` | 不融合（**名单里有，但没有 kernel 路径**） |
| `Conv → Cast → Relu` | 融合（Cast 先被删） |
| `Conv` 输出被两个节点用 | 不融合（另一消费者拿不到数据） |
| `Conv → Add → Relu` | 三级一起融 |

`Elu` 那条：**名字在白名单里也未必生效，还要有 kernel 实现。**

### 5.5 该用哪个

| 目标 | 选择 |
|---|---|
| 模型要分发 / 上不同硬件 | onnxsim |
| 在这台机器上跑最快 | 先用 onnxsim 清理，再让 ORT 优化 |
| 要给 TVM / MNN / onnx-mlir | 只用 onnxsim |
| 调试自定义算子 | `ORT_DISABLE_ALL`，别让融合把节点弄没了 |
| 量化模型 | 两者都别指望 |

---

## 六、API 的两个坑

```python
# 坑 1：返回 tuple，不是 model
r = onnxsim.simplify(m)      # ❌ 拿到 tuple，r.graph 会 AttributeError
m, ok = onnxsim.simplify(m)  # ✅

# 坑 2：迭代次数只能走环境变量
os.environ["ONNXSIM_FIXED_POINT_ITERS"] = "1"
```

`ok` 的含义也要清楚：

| `check_n` | 做什么 |
|---|---|
| 0（默认） | 只跑 `onnx.checker` 校验结构，**不验数值** |
| >0 | 随机生成 N 组输入，原模型 vs 简化模型真实对拍 |

实验中 `ok=True` 和 `bit-equal=True` 是两套独立机制，同时通过只是巧合。

---

## 七、运行方式

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 e1.py    # 最小化简：三类动作
python3 e2.py    # 迭代到不动点
python3 e3.py    # 四条边界
python3 e4.py    # 与 ORT 的分工
```

本机实测环境（2026-11）：

```
Python 3.11.14
onnxsim 0.8.1
onnx 1.23.2
onnxruntime 1.31.1
```

完整输出存档在 `e3_output.txt`、`e4_output.txt`。

> **版本敏感**：onnxsim 0.7.3 与 0.8.1 在迭代次数上表现不同（3.4 节）。
> 换版本重跑时，先确认 v0.7.3 那组数字是否还成立。

---

## 八、学完应该能回答

1. onnxsim 和 ORT 的图优化，产物有什么区别？为什么这个区别决定了能不能分发？
2. `deep_bn` 为什么不能一轮融完？画出每轮的依赖关系。
3. 为什么结果已经正确，却还在报「未收敛」？
4. 什么时候可以用 `overwrite_input_shapes`？
5. 你的模型里有自定义算子，onnxsim 会对它做什么？为什么"不做"是正确行为？
6. `ok=True` 能证明化简前后等价吗？
7. 已经跑了 onnxsim，还需要开 `ORT_ENABLE_ALL` 吗？
8. 为什么 `Conv → Elu` 在白名单里却不融合？
