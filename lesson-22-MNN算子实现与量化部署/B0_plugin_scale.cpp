// B0_plugin_scale.cpp
// Lesson 22 实验B: 手写 MNN 最小自定义算子 PluginScale
//
// 目标: 在不改 MNN 源码的前提下, 注册一个自己的算子, 让 MNN 推理时执行它。
// 算子定义: y = x * scale + bias  (scale 和 bias 都用 attr 从模型参数传入)
//
// 这个文件回答一个问题: "MNN 是怎么支持自定义算子的?"
// 答: 三段式注册 (和内置算子同一套骨架):
//   ① InferShapeKernel  (定义输出 shape, 挂在 shape 阶段)
//   ② CPUComputeKernel  (定义计算, 挂在 backend 执行阶段)
//   ③ 两个注册宏         (把类型名 "PluginScale" 绑定到上面两个类)
// 运行时 MNN 用类型名查全局注册表, 找到 kernel 就执行 → 这就是"插件"的本质。
//
// 编译: bash B0_build.sh

#include <MNN/expr/Expr.hpp>                   // 表达式 API: Expr, Variable
#include <MNN/expr/ExprCreator.hpp>            // _Input 等算子工厂
#include <MNN/plugin/PluginKernel.hpp>         // CPUComputeKernel + REGISTER_PLUGIN_COMPUTE_KERNEL
#include <MNN/plugin/PluginShapeInference.hpp> // InferShapeKernel + REGISTER_PLUGIN_OP
#include "MNN_generated.h"                     // OpT/PluginT/OpType_Plugin 的完整定义
#include <iostream>

using namespace MNN;          // Interpreter 等
using namespace MNN::Express; // Expr, Variable, VARP
using namespace MNN::plugin;  // InferShapeKernel, CPUComputeKernel, 注册宏

// ═══════════════════════════════════════════════════════════════════
// ① Shape 阶段: 告诉 MNN 这个算子的输出长什么样
//    MNN 推理前先算好所有中间 tensor 的 shape, 再分配内存
// ═══════════════════════════════════════════════════════════════════
class PluginScaleInferShape : public InferShapeKernel
{
public:
    bool compute(InferShapeContext *ctx) override
    {
        MNN_CHECK(ctx->inputs().size() == 1, "PluginScale needs 1 input");
        MNN_CHECK(ctx->outputs().size() == 1, "PluginScale needs 1 output");

        const auto &x = ctx->input(0)->buffer(); // 输入 tensor 的 buffer 描述
        auto &output = ctx->output(0)->buffer(); // 输出 tensor 的 buffer 描述

        // 逐元素算子: 输出和输入同 shape、同类型
        output.dimensions = x.dimensions;
        for (int i = 0; i < x.dimensions; ++i)
        {
            output.dim[i].extent = x.dim[i].extent;
        }
        output.type = x.type;
        return true;
    }
};

// ═══════════════════════════════════════════════════════════════════
// ② Backend 阶段: 真正的计算
//    三段式: init (构造时, 查注册表后) → resize (shape 定好后) → compute (每帧执行)
// ═══════════════════════════════════════════════════════════════════
class PluginScaleKernel : public CPUComputeKernel
{
public:
    // init: 算子构造时调用, 读取 attr (scale 和 bias 从模型参数来)
    bool init(CPUKernelContext *ctx) override
    {
        scale_ = 1.0f;
        bias_ = 0.0f;
        if (ctx->hasAttr("scale"))
        {
            scale_ = ctx->getAttr("scale")->f(); // flatbuffer Attribute 的 float 字段
        }
        if (ctx->hasAttr("bias"))
        {
            bias_ = ctx->getAttr("bias")->f();
        }
        return true;
    }
    // resize: shape 定好后调用，算好元素个数
    bool resize(CPUKernelContext *ctx) override
    {
        const auto &x = ctx->input(0)->buffer();
        count_ = 1;
        for (int i = 0; i < x.dimensions; ++i)
        {
            count_ *= x.dim[i].extent;
        }
        return true;
    }
    // compute: 每帧执行, 真正的 y = x * scale + bias
    bool compute(CPUKernelContext *ctx) override
    {
        const auto &x = ctx->input(0)->buffer();
        auto &output = ctx->output(0)->buffer();

        // MNN 里 host 指针就是 CPU 内存 (后端是 CPU 才有这个)
        const float *x_data = reinterpret_cast<const float *>(x.host);
        float *out_data = reinterpret_cast<float *>(output.host);

        for (int i = 0; i < count_; ++i)
        {
            out_data[i] = x_data[i] * scale_ + bias_;
        }
        return true;
    }

private:
    float scale_ = 1.0f;
    float bias_ = 0.0f;
    int count_ = 0;
};

// ═══════════════════════════════════════════════════════════════════
// ③ 注册: 把字符串类型名绑定到上面的两个类
//    REGISTER_PLUGIN_OP               → 注册到 shape 阶段的注册表
//    REGISTER_PLUGIN_COMPUTE_KERNEL   → 注册到 backend 阶段的注册表
//    宏内部: 匿名命名空间里定义一个静态对象, 构造时把工厂函数塞进注册表
//    静态对象在"链接进主程序"时自动初始化 → 注册自动完成
// ═══════════════════════════════════════════════════════════════════
REGISTER_PLUGIN_OP(PluginScale, PluginScaleInferShape);
REGISTER_PLUGIN_COMPUTE_KERNEL(PluginScale, PluginScaleKernel);

// ═══════════════════════════════════════════════════════════════════
// 主程序: 用 Express API 构造一个"插件算子"的图, 跑推理, 验证数值
// 关键: 直接用 OpT + PluginT 构造 OpType_Plugin, 不经过 ONNX 转换
// ═══════════════════════════════════════════════════════════════════
int main()
{
    // ── ① 构造输入 x = [1, 2, 3, 4] ──
    // VARP 是 shared_ptr<Variable> 的别名，理解为"一个张量变量"。
    VARP x = _Input({1, 4}, Express::NCHW); // 拿到这块内存的可写指针
    auto xPtr = x->writeMap<float>();
    for (int i = 0; i < 4; ++i)
    {
        xPtr[i] = i + 1.0f;
    }

    // ── ② 构造 plugin 算子: OpType_Plugin + Plugin 参数 ──
    //    这就是"自定义算子"在模型里的表示: 一个 Plugin 类型的 op, 带 type 名字 + attrs
    //    三层嵌套结构: OpT(外层: 什么算子) → PluginT(中层: 哪个插件) → AttributeT(内层: 参数)
    std::unique_ptr<OpT> pluginOp(new OpT); // 外层 OpT: 一张"算子档案卡"
    pluginOp->type = OpType_Plugin;         //   标记"这是插件算子"

    PluginT *pluginParam = new PluginT; // 中层 PluginT: 这个插件具体是谁
    pluginParam->type = "PluginScale";  //   ← 注册表里的名字, 运行时按它查 kernel

    // scale attr
    AttributeT *scaleAttr = new AttributeT;
    scaleAttr->key = "scale";
    scaleAttr->f = 3.0f; // scale = 3.0
    pluginParam->attr.emplace_back(scaleAttr);

    // bias attr
    AttributeT *biasAttr = new AttributeT;
    biasAttr->key = "bias";
    biasAttr->f = 0.5f; // bias = 0.5
    pluginParam->attr.emplace_back(biasAttr);

    pluginOp->main.type = OpParameter_Plugin;
    pluginOp->main.value = pluginParam;

    // ── ③ 建图: y = PluginScale(x) ──
    VARP y = Variable::create(Expr::create(pluginOp.get(), {x})); // 把"算子 + 它的输入"组合成一个表达式节点（类似"y 是由 PluginScale 算出来的，输入是 x"）。

    // ── ④ 推理 (调用 readMap 触发实际执行) ──
    auto yInfo = y->getInfo();       // 触发 shape 推断 → 调用 PluginScaleInferShape::compute
    auto yPtr = y->readMap<float>(); // 触发执行 → 调用 PluginScaleKernel::compute

    // ── ⑤ 验证: y = x*3 + 0.5 → 期望 [3.5, 6.5, 9.5, 12.5] ──
    std::cout << "PluginScale 输出 shape: " << yInfo->dim.size() << " 维 [";
    for (auto d : yInfo->dim)
        std::cout << d << " ";
    std::cout << "]\n";

    float expected[4] = {3.5f, 6.5f, 9.5f, 12.5f};
    bool ok = true;
    for (int i = 0; i < 4; ++i)
    {
        std::cout << "  y[" << i << "] = " << yPtr[i]
                  << "  (期望 " << expected[i] << ")\n";
        if (std::abs(yPtr[i] - expected[i]) > 1e-5)
            ok = false;
    }
    std::cout << (ok ? "[PASS] PluginScale 数值正确!" : "[FAIL] 数值不对!")
              << std::endl;
    return ok ? 0 : 1;
}