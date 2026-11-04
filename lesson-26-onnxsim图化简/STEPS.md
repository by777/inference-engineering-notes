# Lesson 26 实验步骤

每个实验独立可跑。**先看「预期」，再跑，最后对差异。**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -c "import onnxsim, onnx, onnxruntime as ort; print(onnxsim.__version__, onnx.__version__, ort.__version__)"
```

把版本号记下来。本课所有数字都绑定版本，换版本结论可能变。

---

## 实验 1：最小化简

**目的**：一张人造图里同时看到「消除 / 常量折叠 / 融合」三类动作。

```bash
python3 e1.py
```

**预期**

```
原始 : ['Identity', 'Cast', 'MatMul', 'Add', 'Mul', 'Mul', 'Div', 'Identity'] 4 inits
返回 : tuple 2
简化 : ['Gemm', 'Mul', 'Div'] 3 inits
ok   : True
k 的值: [[6.0]]
max|diff|: 0.0  bit-equal: True
```

**要盯的三处**

| 现象 | 含义 |
|---|---|
| `返回 : tuple 2` | `simplify()` 返回 `(model, ok)`，不是 model |
| `k = 6.0`，4→3 inits | `Mul(2.0,3.0)` 被常量折叠 |
| `MatMul+Add → Gemm` | 融合，且产物仍是 `ai.onnx` 标准算子 |

`Mul(x,k)` 和 `Div(·,k)` 没被约掉，是对的 —— 浮点下不等价。

---

## 实验 2：迭代到不动点

**目的**：证明「一遍不够」是真的，并说明原因。

```bash
python3 e2.py
```

**预期**

```
原始 : ['Conv','Mul','BatchNormalization','Mul','BatchNormalization','Mul','BatchNormalization'] 8 inits

iters  节点                                              inits  已收敛
    1  ['Conv','BatchNormalization','BatchNormalization','BatchNormalization']  6    否
    2  ['Conv']                                          2      否
    3  ['Conv']                                          2      是

手动循环（每次 1 轮）:
  第 1 次: ['Conv','BN','BN','BN']
  第 2 次: ['Conv']
  第 3 次: ['Conv']
```

**为什么一遍不够**

`fuse_bn_into_conv` 要求 BN 的 `scale` 是常量。但 `sc1` 是 `Mul(a,b)` 的输出，是中间张量。融合先跑，看到 `sc1` 算不出新权重就跳过；常量折叠后跑，把 `sc1` 变成 initializer —— 但这一轮的融合已经结束了。

折叠的产出只能被**下一轮**的融合看到。

**两个注意点**

1. **结果正确 ≠ 已收敛**：`iters=3` 图已经对了，但可能还在报 WARNING。判据是「这次跑完和上次一样」，需要一轮确认。
2. **警告抓不到**：它由 C++ 直接写 fd 1，Python 的 `redirect_stdout`/`redirect_stderr` 都拦不住，必须用 `subprocess` 读子进程输出。

**版本差异**：onnxsim 0.7.3 需要 5 轮才收敛、且 `iters=1` 能融一层；0.8.1 只要 3 轮、`iters=1` 一层都没融。**别硬编码轮数。**

---

## 实验 3：四条边界

**目的**：区分「工具漏了」和「工具故意不做」。

```bash
python3 e3.py
```

**预期**

| 边界 | 化简前 | 化简后 | 判断 |
|---|---|---|---|
| ① 动态 shape | `Shape,Gather,Concat,Reshape,Identity` | `Shape,Gather,Concat,Reshape` | 保留动态 |
| ② overwrite | 同上 | `Reshape` | 焊死 |
| ③ 自定义算子 | `Identity,MyAdd,Mul,Div,Identity` | `MyAdd,Mul,Div` | 不穿透 |
| ④ QDQ | `Q,DQ,Mul,Div` | `Q,DQ,Mul,Div` | 完全不碰 |

**逐条要点**

**① 动态**：第 0 维仍是 `'batch'`，batch=1/3/8 都跑通。只有 `Identity` 被删。

**② overwrite**：第 0 维变成 `1`，整条 `Shape` 链消失，`batch=3` 报 `INVALID_ARGUMENT`。这是"图里写死 1、你给 3"，不是算错。

**③ 自定义算子**：`MyAdd` 前后的 `Mul`/`Div` 不动。因为 onnxsim 不知道 `MyAdd` 会不会偷偷改 `k`。不认识的算子 = 一堵墙，这是保守但正确的。

**④ QDQ**：一个节点都不动。Q 是有损的（`0.1234 → int8 → 0.1`），约掉就把量化模型变回浮点模型了。

---

## 实验 4：与 ORT 的分工

**目的**：同一张图交给两条路，对比产物差异，理解 `nchwc` 从哪来。

```bash
python3 e4.py
```

**预期**

```
Conv+BN+Relu
  原始       ['ai.onnx:Conv', 'ai.onnx:BatchNormalization', 'ai.onnx:Relu']
  onnxsim    ['ai.onnx:Conv', 'ai.onnx:Relu']
  ORT(ALL)   ['com.microsoft.nchwc:Conv', 'com.microsoft.nchwc:ReorderOutput']
  sim→ORT    ['com.microsoft.nchwc:Conv', 'com.microsoft.nchwc:ReorderOutput']
  三条路数值都 bit-equal=True

MatMul+Add
  原始       ['ai.onnx:MatMul', 'ai.onnx:Add']
  onnxsim    ['ai.onnx:Gemm']
  仅 ORT     ['ai.onnx:Gemm']
  sim→ORT    ['ai.onnx:Gemm']
```

**三个结论**

1. **都能融合，但产物不同域**：onnxsim 给标准算子（可分发），ORT 给私有算子（绑环境）。
2. **收益重叠，不是叠加**：`sim→ORT` 与只跑 ORT 结果相同。onnxsim 先合成了 `Gemm`，ORT 就无从下手了。
3. **融合的物理形态**：`Relu` 变成 `Conv` 的 `activation` 属性。

```
com.microsoft.nchwc:Conv
  inputs = ['X', 'reorder', 'ConvBnFusion_BN_B_bb']
  attr activation = b'Relu'
```

**为什么 onnxsim 不产出私有算子**

融合算子必须是已注册的实体。造一个不存在的：

```
com.microsoft:FusedNotExist  ->  FAIL: is not a registered function/op
```

`nchwc` 依赖 CPU EP 里的对应 kernel。换 GPU / ARM / ORT 版本，kernel 不存在，模型直接加载失败。onnxsim 保持标准算子，是为了维持"通用化简器"的定位。

---

## 决策速查

| 目标 | 选择 |
|---|---|
| 模型要分发 / 上不同硬件 | onnxsim |
| 在这台机器上跑最快 | onnxsim 清理，再让 ORT 优化 |
| 要给 TVM / MNN / onnx-mlir | 只用 onnxsim |
| 调试自定义算子 | `ORT_DISABLE_ALL` |
| 量化模型 | 用专门的量化工具 |
