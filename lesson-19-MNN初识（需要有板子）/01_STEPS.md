# MNN 环境搭建 STEPS（已验证通过 2026-08-17，MNN 3.6.1）

## 1. 克隆源码（和 tvm-src 同级）

```bash
cd $REPO
git clone --depth 1 https://github.com/alibaba/MNN.git mnn-src
```

## 2. 编译 C++ 引擎（libMNN.so + MNNConvert 等工具，5-10 分钟）

```bash
cd mnn-src && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_CONVERTER=ON
make -j$(nproc)
```

产物：
- `build/libMNN.so`  → 推理引擎
- `build/MNNConvert` → 命令行模型转换器（ONNX/Caffe→MNN）

## 3. 编译 Python 绑定（⚠️ 走 pymnn/pip_package，不要用 MNN_BUILD_PYTHON）

> `MNN_BUILD_PYTHON=ON` 只对 Android 生效，PC 上无效（已验证）。
> 官方路径：`pymnn/pip_package` 的 `build_deps.py` + `setup.py install`。
> 必须装进 lesson-13 的 venv（不要 sudo 装系统目录，且系统 python3.10 和你 venv 的 3.11 不一致）。

```bash
# 3.1 先构建 C++ 依赖（会重新编译 libMNN 相关）
cd $REPO/mnn-src/pymnn/pip_package
python3 build_deps.py

# 3.2 激活 venv + 补 setuptools（uv 创建的 venv 默认没有 pip/setuptools）
cd $REPO
source lesson-13-AI编译器入门（TVM）/.venv/bin/activate
$HOME/miniforge3/bin/uv pip install --python "$REPO/lesson-13-AI编译器入门（TVM）/.venv/bin/python" setuptools

# 3.3 在 venv 里安装 MNN
cd mnn-src/pymnn/pip_package
python3 setup.py install
```

## 4. 验证

```bash
# 先建 venv 软链接（一劳永逸，之后直接 source .venv 即可）
cd $REPO/lesson-19-MNN初识
ln -sfn ../lesson-13-AI编译器入门（TVM）/.venv .venv

source .venv/bin/activate
python3 -c "import MNN; print('MNN OK, 版本:', MNN.version())"
# 预期: MNN OK, 版本: 3.6.1（前几行 CPU Group 是硬件探测，正常）
```

## 5. 顺带获得的工具（setup.py 装进 venv/bin）

```bash
which mnnconvert   # ONNX/Caffe → .mnn 命令行转换器
which mnnquant     # 模型量化工具
```

## 每日使用（新终端，只需一行）

```bash
cd $REPO/lesson-19-MNN初识
source .venv/bin/activate
```

## 踩坑记录

| # | 错误 | 原因 | 解决 |
|---|------|------|------|
| 1 | `No module named 'MNN'`（PYTHONPATH=build） | Python 绑定不在 build 根目录，走的是 setuptools 安装 | 用 `pymnn/pip_package` 的 `setup.py install` |
| 2 | `Permission denied: /usr/local/lib/python3.10/...` | 用了系统 python3.10 + sudo 目录 | 在 lesson-13 venv（python3.11）里装 |
| 3 | `No module named 'setuptools'` | uv 创建的 venv 没带 setuptools | `uv pip install --python <venv绝对路径> setuptools` |
| 4 | `uv: No virtual environment found for path .venv/bin/python` | uv 的相对路径找不到 venv | 用 venv 的**绝对路径** |

## 环境结构

```
onnx_runtime_ops/
├── mnn-src/                    # MNN 源码
│   ├── build/libMNN.so         # C++ 推理引擎
│   ├── build/MNNConvert        # 命令行转换器
│   └── pymnn/pip_package/      # Python 包源码（setup.py 装进 venv）
├── lesson-19-MNN初识/
│   └── STEPS.md                # 本文档
└── lesson-13-AI编译器入门（TVM）/.venv/   # MNN 3.6.1 装在这
```
