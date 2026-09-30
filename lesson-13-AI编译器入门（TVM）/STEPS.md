# Lesson 13: AI 编译器入门（TVM）

## 课程目标

学完本课你应该能：

1. **理解 AI 编译器的三层架构**：前端（图 IR 翻译）→ 中端（pass 优化）→ 后端（代码生成）
2. **用 TVM 编译并执行 ONNX 模型**，且与 ONNX Runtime 结果一致
3. **用 TE（Tensor Expression）手写算子**，理解"计算与调度分离"的编译思想
4. **用 Relay pass 观察图优化**（算子融合），知道 `opt_level=3` 背后发生了什么
5. **独立搭建 TVM 环境**（换机器时按本 README 可复现）

---

## 一、环境搭建

### 1.1 为什么不用 pip/conda 装 TVM

| 方式 | 结果 | 原因 |
|------|------|------|
| `pip install apache-tvm` | ❌ import 报错 | 0.25.0 把 ffi 拆成独立包，但 `apache-tvm-ffi` 缺 `json::Stringify` 符号 |
| `conda install -c conda-forge tvm` | ❌ 找不到包 | 清华镜像没有 tvm |
| **源码编译 v0.18.0** | ✅ 成功 | 旧版自包含，无拆分问题 |

**结论**：源码编译稳定版 v0.18.0。

### 1.2 前置条件

```bash
# 系统需有 LLVM 14（代码生成后端）
llvm-config --version        # 输出 14.x

# uv（虚拟环境管理，装在 miniforge 里）
$HOME/miniforge3/bin/uv --version  # 输出 0.9.x

# Python 3.11
python3 --version
```

### 1.3 编译 TVM

```bash
cd onnx_runtime_ops/
git clone --recursive https://github.com/apache/tvm tvm-src
cd tvm-src

git checkout v0.18.0
git submodule update --init --recursive

mkdir -p build && cd build
cp ../cmake/config.cmake .
echo "set(USE_LLVM /usr/bin/llvm-config)" >> config.cmake    # 后端用 LLVM 14
echo "set(USE_GRAPH_EXECUTOR ON)" >> config.cmake            # 图执行器（推理用）
echo "set(CMAKE_BUILD_TYPE Release)" >> config.cmake

cmake ..
make -j8        # 首次约 30-60 分钟，出现 [100%] Built target tvm 即成功
```

### 1.4 拷贝产物 + 建虚拟环境

```bash
# 拷贝 .so 到 tvm-bin（类似 ort-bin 的本地依赖目录）
cd onnx_runtime_ops/
mkdir -p tvm-bin
cp tvm-src/build/libtvm.so tvm-src/build/libtvm_runtime.so tvm-bin/

# 建 venv 并装 Python 依赖
uv venv --python 3.11
source .venv/bin/activate
$HOME/miniforge3/bin/uv pip install --python .venv/bin/python \
    decorator psutil scipy attrs typing-extensions cloudpickle packaging tqdm
```

### 1.5 每日使用（新终端必做）

```bash
cd "$REPO/lesson-13-AI编译器入门（TVM）"
source .venv/bin/activate
export PYTHONPATH="$REPO/tvm-src/python:$REPO/tvm-bin"

# 验证
python -c "import tvm; print('TVM:', tvm.__version__)"   # 应输出 0.18.0
```

### 1.6 冒烟测试

```python
# 编译 y[i] = x[i] + 1 并执行，验证 LLVM 代码生成链路
import tvm, numpy as np
from tvm import te

n = 4
x = te.placeholder((n,), name='x')
y = te.compute((n,), lambda i: x[i] + 1, name='y')
s = te.create_schedule(y.op)
f = tvm.build(s, [x, y], target='llvm')          # LLVM 代码生成

x_tvm = tvm.nd.array(np.array([1,2,3,4], dtype='float32'))
y_tvm = tvm.nd.empty((n,), dtype='float32')
f(x_tvm, y_tvm)
assert y_tvm.numpy().tolist() == [2,3,4,5]        # 执行验证
print('OK')
```

> ⚠️ numpy 数组必须 `dtype='float32'`（numpy 默认 float64，会报 dtype 不匹配）

---

## 二、踩坑记录

| # | 错误 | 原因 | 解决 |
|---|------|------|------|
| 1 | `undefined symbol: _ZN3tvm3ffi4json9Stringify...` | `apache-tvm 0.25` 与 `apache-tvm-ffi 0.1.13` 版本不匹配（上游打包 bug） | 源码编译 v0.18.0 |
| 2 | `No module named 'onnx.mapping'` | onnx 1.16+ 移除了 `mapping` 模块，TVM 0.18 前端还在用 | `uv pip install onnx==1.15.0` |
| 3 | `Command 'uv' not found` | uv 不在 PATH，装在 miniforge | 用绝对路径 `$HOME/miniforge3/bin/uv` |
| 4 | `dtype is expected to be float32` | numpy 数组默认 float64，与 TVM placeholder 不符 | 显式 `dtype='float32'` |

---

## 三、环境结构

```
onnx_runtime_ops/
├── ort-bin/                # ONNX Runtime 本地依赖
├── tvm-src/                # TVM 源码（编译产物在 tvm-src/build/）
├── tvm-bin/                # TVM 本地依赖（libtvm.so + libtvm_runtime.so）
└── lesson-13-AI编译器入门（TVM）/
    ├── .venv/              # Python 3.11 虚拟环境
    ├── mlp3.onnx           # 演示模型（3 层 MLP）
    ├── 01_te_intro.py      # 实验 1: TE 手写算子
    ├── 02_from_onnx.py     # 实验 2: ONNX → Relay → 编译 → 执行
    ├── 03_relay_pass.py    # 实验 3: Relay pass 内省
    └── README.md           # 本文档
```