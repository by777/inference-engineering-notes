"""实验 4：onnxsim 与 ORT 的分工 —— 同一张图，两种产物

目的
    前三轮实验只看了 onnxsim。这一轮把它和 ORT 运行时的图优化放在一起,
    回答一个实践问题：既然是同一套融合规则，为什么还要分两层？

    对比三条路：
      A. 只跑 onnxsim
      B. 只让 ORT 优化（SessionOptions + optimized_model_filepath 落盘）
      C. 先 onnxsim 再 ORT

核心结论（实测）
    A 的产物是 ai.onnx 标准算子，能分发
    B 的产物是 com.microsoft.nchwc 私有算子，绑环境
    C 与 B 结果相同 —— 收益会重叠，不是叠加

为什么这个区别重要
    ORT 的融合会改变数据布局（NCHW -> NCHWc），这需要对应的 kernel 才跑得起来。
    onnxsim 保持标准算子，所以它不能做布局变换，但产物哪里都能用。

运行
    python3 e4.py
"""

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh


def f32(v, n):
    return nh.from_array(np.asarray(v, np.float32), n)


def vi(name, shape):
    return h.make_tensor_value_info(name, T.FLOAT, shape)


def op_names(m):
    """带域名列出节点，自定义算子要能看出是哪个域。"""
    return [f"{n.domain or 'ai.onnx'}:{n.op_type}" for n in m.graph.node]


def build_conv_bn_relu(tag="conv_bn_relu"):
    """Conv -> BN -> Relu，最经典的融合目标。"""
    Wc = f32(np.random.randn(8, 3, 3, 3) * 0.1, "Wc")
    nodes = [
        h.make_node(
            "Conv", ["X", "Wc"], ["cv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        ),
        h.make_node("BatchNormalization", ["cv", "s", "bb", "m", "v"], ["bn"]),
        h.make_node("Relu", ["bn"], ["Y"]),
    ]
    inits = [
        Wc,
        f32(np.random.rand(8) + 0.5, "s"),
        f32(np.random.randn(8) * 0.1, "bb"),
        f32(np.random.randn(8) * 0.1, "m"),
        f32(np.random.rand(8) + 0.1, "v"),
    ]
    g = h.make_graph(
        nodes, tag, [vi("X", [1, 3, 8, 8])], [vi("Y", [1, 8, 8, 8])], inits
    )
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 15)])
    m.ir_version = 10
    onnx.save(m, f"{tag}.onnx")
    return f"{tag}.onnx"


def build_matmul_add(tag="matmul_add"):
    """MatMul + Add，另一个常见的融合目标（合成 Gemm）。"""
    W = f32(np.random.randn(4, 4) * 0.1, "W")
    b = f32(np.random.randn(4) * 0.1, "b")
    nodes = [
        h.make_node("MatMul", ["X", "W"], ["mm"]),
        h.make_node("Add", ["mm", "b"], ["Y"]),
    ]
    g = h.make_graph(nodes, tag, [vi("X", [1, 4])], [vi("Y", [1, 4])], [W, b])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.save(m, f"{tag}.onnx")
    return f"{tag}.onnx"


def run_onnxsim(src, dst, **kw):
    """onnxsim 化简并存盘，返回节点列表。"""
    m, _ = onnxsim.simplify(onnx.load(src), **kw)
    onnx.save(m, dst)
    return op_names(m)


def run_ort(src, dst, level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
    """让 ORT 优化并把优化后的图落盘。

    optimized_model_filepath 是关键：它把 ORT 内部优化后的图序列化出来，
    没有它你只能看到耗时，看不到图变成了什么样。
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    so.log_severity_level = 3  # 只留 Error，压掉 nchwc 的警告
    so.optimized_model_filepath = dst
    ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
    return op_names(onnx.load(dst))


def run_infer(path, feeds):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s.run(None, feeds)[0]


def header(t):
    print()
    print("=" * 68)
    print(t)
    print("=" * 68)


def compare_model(name, builder, feeds):
    src = builder()
    header(f"模型：{name}")

    print(f"  {'原始':<12} {op_names(onnx.load(src))}")

    a = run_onnxsim(src, f"{name}_sim.onnx")
    print(f"  {'onnxsim':<12} {a}")

    b = run_ort(src, f"{name}_ort.onnx")
    print(f"  {'ORT(ALL)':<12} {b}")

    c = run_ort(f"{name}_sim.onnx", f"{name}_sim_ort.onnx")
    print(f"  {'sim→ORT':<12} {c}")

    # 数值三方对拍：化简不能改语义
    ref = run_infer(src, feeds)
    for tag, p in [("onnxsim", f"{name}_sim.onnx"), ("ORT", f"{name}_ort.onnx"),
                   ("sim→ORT", f"{name}_sim_ort.onnx")]:
        got = run_infer(p, feeds)
        print(f"    {tag:<8} max|diff|={np.abs(got - ref).max():.3e}", end="")
        print(f"  bit-equal={np.array_equal(got.view(np.int32), ref.view(np.int32))}")
    print()


def show_fused_attrs():
    """把融合后的节点属性打印出来，看激活是怎么被吸收的。"""
    header("融合后的节点长什么样")
    m = onnx.load("conv_bn_relu_ort.onnx")
    for n in m.graph.node:
        print(f"  {n.domain}:{n.op_type}")
        print(f"    inputs = {list(n.input)}")
        for attr in n.attribute:
            val = onnx.helper.get_attribute_value(attr)
            s = str(val)
            print(f"    attr {attr.name:<16} = {s[:60]}")
    print()
    print("  → activation 变成了 Conv 的一个属性，Relu 节点消失了。")
    print("    这就是「融合」的物理形态：下游语义变成上游的一个字段。")


def compare_overlap():
    """证明收益会重叠：onnxsim 做过的事 ORT 就不做。"""
    header("收益重叠：onnxsim 先做过，ORT 就无事可做")
    print(f"  {'MatMul+Add 原始':<20} {op_names(onnx.load('matmul_add.onnx'))}")
    print(f"  {'仅 onnxsim':<20} {op_names(onnx.load('matmul_add_sim.onnx'))}")
    print(f"  {'仅 ORT':<20} {op_names(onnx.load('matmul_add_ort.onnx'))}")
    print(f"  {'sim→ORT':<20} {op_names(onnx.load('matmul_add_sim_ort.onnx'))}")
    print()
    print("  Conv+BN+Relu 同理：最终都收敛到同一个结果，")
    print("  叠加不会产生额外收益 —— 该合的已经合完了。")


def main():
    rng = np.random.RandomState(0)
    compare_model(
        "conv_bn_relu",
        build_conv_bn_relu,
        {"X": rng.randn(1, 3, 8, 8).astype(np.float32)},
    )
    compare_model(
        "matmul_add",
        build_matmul_add,
        {"X": rng.randn(1, 4).astype(np.float32)},
    )
    show_fused_attrs()
    compare_overlap()

    header("结论")
    print("  1. 两者都能融合，但产物不同域：")
    print("     onnxsim -> ai.onnx 标准算子，可分发")
    print("     ORT     -> com.microsoft.* 私有算子，绑环境")
    print()
    print("  2. 收益重叠，不是叠加。先 sim 后 ORT 与只跑 ORT 结果相同。")
    print()
    print("  3. 融合产物必须是对应后端已注册的 kernel，")
    print("     否则模型直接加载失败：com.microsoft:FusedNotExist 会报")
    print("     'is not a registered function/op'。")


if __name__ == "__main__":
    main()
