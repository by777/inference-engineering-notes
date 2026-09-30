# Lesson 22 实验B: 手写 MNN 最小自定义算子 PluginScale

## 文件清单（全部跑通）

| 文件 | 作用 | 怎么验证的 |
|---|---|---|
| `B0_plugin_scale.cpp` | 手写 PluginScale 算子（InferShape + ComputeKernel + 2 注册宏）+ Express API 构造图 + 数值验证 | 输出 `[PASS]` |
| `B0_build.sh` | 编译 + 运行脚本（含 3 个 include 路径 + 2 个 lib 路径） | `sh B0_build.sh` 一键通过 |

## 目标

在不改 MNN 源码的前提下，注册一个自己的算子 `PluginScale`，让 MNN 推理时执行它。
算子定义：`y = x * scale + bias`（scale、bias 都从模型参数 attr 传入）。

## 核心机制：三段式注册（和内置算子同一套骨架）

MNN 内置算子（比如实验A 的 `ConvolutionDepthwise`）和插件算子，走的都是**同一套三段式**：

```
① InferShapeKernel    定义输出 shape，挂在 shape 阶段（推理前算好内存布局）
② CPUComputeKernel    定义计算，挂在 backend 执行阶段
③ 两个注册宏           把字符串类型名 "PluginScale" 绑定到上面两个类
```

运行时 MNN 拿算子节点的 `type` 字符串去查**全局注册表**，查到工厂函数就 new 出 kernel 执行。

### ① Shape 阶段：InferShapeKernel

```cpp
class PluginScaleInferShape : public InferShapeKernel {
    bool compute(InferShapeContext* ctx) override {
        // 输出和输入同 shape、同类型（逐元素算子，形状不变）
        const auto& x = ctx->input(0)->buffer();
        auto& output = ctx->output(0)->buffer();
        output.dimensions = x.dimensions;
        for (int i = 0; i < x.dimensions; ++i)
            output.dim[i].extent = x.dim[i].extent;
        output.type = x.type;
        return true;
    }
};
```

回答一个问题：**给定输入 shape，输出 shape 是什么？** 引擎要先知道每个中间张量多大，才能分配内存。

### ② Backend 阶段：CPUComputeKernel（三个生命周期方法）

```cpp
class PluginScaleKernel : public CPUComputeKernel {
    bool init(CPUKernelContext* ctx)   { /* 构造时: 读 attr 里的 scale/bias */ }
    bool resize(CPUKernelContext* ctx) { /* shape 定好后: 算元素个数 count_ */ }
    bool compute(CPUKernelContext* ctx){ /* 每帧执行: y = x*scale + bias */ }
};
```

三个方法对应三个阶段：
- `init`：算子对象创建时调一次，从 `ctx->getAttr("scale")->f()` 读出模型里存的参数
- `resize`：shape 定好后调，把元素个数累乘进 `count_`，避免 compute 每帧重算形状
- `compute`：每帧推理都调，`x.host` / `output.host` 是 CPU 内存指针，做真正的计算

### ③ 注册宏

```cpp
REGISTER_PLUGIN_OP(PluginScale, PluginScaleInferShape);          // shape 注册表
REGISTER_PLUGIN_COMPUTE_KERNEL(PluginScale, PluginScaleKernel);  // backend 注册表
```

宏展开后是**匿名命名空间里的 static 对象**，构造函数把工厂函数塞进注册表。static 对象在 `main` 之前自动构造 → 自动完成注册，无需手动调用。

## main 里在干什么：手工拼一个算子节点

`_Input` 是 MNN 写好的工厂函数，但 `PluginScale` 是自定义算子，**没有现成工厂函数**，所以得手工把"一个算子节点"的完整结构拼出来。这个结构和实验A 用 `MNNDump2Json` 看到的 `.mnn` 里每个节点**是同一个东西**（OpT 就是那个 JSON 的 C++ 形式）。

三层嵌套结构：

```
OpT（外层: 什么算子）
├─ type = OpType_Plugin        → "这是插件算子"
└─ main（参数联合体 union）
    └─ PluginT（中层: 哪个插件）
        ├─ type = "PluginScale" → 注册表里的名字，运行时靠它查你的实现
        └─ attr = [ ... ]       → 参数列表
            ├─ AttributeT { key="scale", f=3.0 }   ← 内层
            └─ AttributeT { key="bias",  f=0.5 }
```

关键点：
- `PluginT.type = "PluginScale"` 这个字符串，必须和 `REGISTER_PLUGIN_OP(PluginScale, ...)` 宏展开后的 `"PluginScale"` **一致**，运行时才能查到你的实现
- `AttributeT` 里 `f` 存 float（还有 `i`/`b`/`s` 存 int/bool/string），这就是 `init` 里 `getAttr("scale")->f()` 读到的值——参数从"模型"流到"实现"的通路
- `pluginOp->main` 是 union，`main.type = OpParameter_Plugin` 声明现在装的是 PluginT

建图和触发计算：

```cpp
VARP y = Variable::create(Expr::create(pluginOp.get(), {x}));  // 搭图: x → PluginScale → y
auto yInfo = y->getInfo();        // 触发 shape 推断 → PluginScaleInferShape::compute
auto yPtr  = y->readMap<float>(); // 触发执行 → PluginScaleKernel::compute
```

**Express 是惰性求值**：搭图时不算，第一次 `readMap` 要数据才真正执行。`getInfo`/`readMap` 就是触发点，两个类就是这样被调起来的。

## 实测输出（sh B0_build.sh）

```
==== 编译 B0_plugin_scale ====
==== 运行 ====
PluginScale 输出 shape: 2 维 [1 4 ]
  y[0] = 3.5  (期望 3.5)
  y[1] = 6.5  (期望 6.5)
  y[2] = 9.5  (期望 9.5)
  y[3] = 12.5  (期望 12.5)
[PASS] PluginScale 数值正确!
```

## 从这几个结果能确认什么（都是亲手验证的）

1. **插件算子 = 三段式注册**：一个 `InferShapeKernel`（shape）+ 一个 `CPUComputeKernel`（计算）+ 两个宏（绑定名字），和内置算子同一套骨架，只是注册表不一样。
2. **参数传递通路**：模型里的 `AttributeT{key, f}` → `init` 里 `getAttr("scale")->f()`，attr 就是自定义算子的"参数怎么从模型到实现"的答案。
3. **名字是注册表的关键**：`PluginT.type` 字符串 ↔ 注册宏的字符串必须一致，运行时靠它查实现——这和实验A 的 `ConvolutionDepthwise` 类型名靠它找实现是同一机制。
4. **Express 是惰性求值**：搭图不算，`getInfo`/`readMap` 才触发执行。这跟实验A 的 `runSession`（主动跑）是两套 API。

## 踩坑记录（都写进了 B0_build.sh）

1. **libMNN.so 要开 MNN_WITH_PLUGIN**：默认 `OFF`，libMNN.so 里有 `"Plugin is not supported. Please recompile with MNN_WITH_PLUGIN enabled."` 字符串。需 `cmake -DMNN_WITH_PLUGIN=ON` 重编，重编后 `nm -D libMNN.so` 能看到 `ComputeKernelRegistry`/`InferShapeKernelRegister` 符号。
2. **缺 `Tensor_generated.h`**：`PluginContext.hpp` 依赖 flatbuffers 生成头，不在 `include/`，在 `schema/current/`。→ `-I../mnn-src/schema/current`
3. **缺 `flatbuffers/flatbuffers.h`**：上面的生成头又 include 了三方 flatbuffers 头。→ `-I../mnn-src/3rd_party/flatbuffers/include`
4. **`OpT`/`PluginT`/`OpType_Plugin` 未定义**：`Expr.hpp` 只前向声明 `OpT`，不拉完整定义。→ 补 `#include "MNN_generated.h"`（也在 `schema/current/`）
5. **Express 符号未定义**（`_Input`/`Expr::create`/`Variable::create`）：Express 是单独的 `libMNN_Express.so`（在 `build/express/` 子目录，不在 `build/` 根）。→ 加 `-L../mnn-src/build/express -lMNN_Express`，运行时 `LD_LIBRARY_PATH` 也要包含 `build/express`

## 为什么不能用 ONNX 转换得到 plugin 算子（读源码确认）

实验A 走的是"ONNX → MNN 转换器"路线。但 plugin 算子**走不通这条路**：

- 读 `tools/converter/source/onnx/onnxOpConverter.cpp`，MNN 转换器对不认识的 ONNX 节点（含 Custom 节点）统一走 `DefaultonnxOpConverter`，转成 `OpType_Extra`（`extra->engine = "ONNX"`），**不会**转成 `OpType_Plugin`。
- 整个 converter 里没有 `OpType_Extra → OpType_Plugin` 的映射。
- 所以 plugin 算子**必须**像本实验这样用 Express API 直接构造，不能从 ONNX 模型转换得到。这是 MNN 的设计使然——官方 `test/plugin/PluginTest.cpp` 也是这么干的。

这就是"自定义算子怎么落地"的答案：**改不了转换器，就绕过转换器，直接用 OpT 结构把节点拼进图里。**
