"""实验 3：onnxsim 的四条边界 —— 它什么时候会「拒绝」动手

目的
    实验 1/2 看的是「它能做什么」。这一轮看反面：「它刻意不做什么」。
    这四个边界都是工程上真会踩到的：

      ① 动态 shape      ——  默认保持动态（正确行为）
      ② overwrite       ——  一旦指定静态，就永久焊死（危险行为）
      ③ 自定义算子      ——  只能清两端，不穿透（保守但正确）
      ④ 量化 QDQ        ——  一个节点都不碰（正确行为）

为什么重要
    「没被优化」有两种含义：
      a) 工具漏了（该优化没优化）—— 损失性能
      b) 工具故意不做（做了会改语义）—— 保护你

    分不清这两者，就会去「修」一个本来正确的行为。

运行
    python3 e3.py
"""

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

# ── 小工具 ──────────────────────────────────────────────────


def f32(v, n):
    """造 float32 常量。"""
    return nh.from_array(np.asarray(v, np.float32), n)


def i64(v, n):
    """造 int64 常量（shape 运算要 int64）。"""
    return nh.from_array(np.asarray(v, np.int64), n)


def vi(name, shape):
    """造一个 float32 输入/输出声明。shape 里可以放字符串表示动态维。"""
    return h.make_tensor_value_info(name, T.FLOAT, shape)


def build(tag, nodes, inputs, outputs, inits, opsets=None):
    """把节点拼成 .onnx 存盘。"""
    opsets = opsets or [h.make_opsetid("", 17)]
    g = h.make_graph(nodes, tag, inputs, outputs, inits)
    m = h.make_model(g, opset_imports=opsets)
    m.ir_version = 10
    onnx.save(m, f"{tag}.onnx")
    return f"{tag}.onnx"


def simp(path, **kw):
    """化简并返回节点类型列表。simplify 返回 tuple，这里解包。"""
    m, ok = onnxsim.simplify(onnx.load(path), **kw)
    return [n.op_type for n in m.graph.node], m, ok


def run(path, feeds):
    """跑一次推理。"""
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s.run(None, feeds)[0]


def op_names(m):
    """列出节点，带域名。

    为什么要带域名：自定义算子打印成 MyAdd 看不出是哪个域，
    而 my_domain:MyAdd 和 ai.onnx:MyAdd 是两回事。
    """
    return [f"{n.domain or 'ai.onnx'}:{n.op_type}" for n in m.graph.node]


def header(t):
    print()
    print("=" * 66)
    print(t)
    print("=" * 66)


# ── 边界 ①：动态 shape 默认被保留 ───────────────────────────
def boundary_dynamic():
    header("① 动态 shape：默认保持动态，不会被焊死")

    # 图的意思是：Y = reshape(X, [X.shape[0], -1])
    # 也就是「保持 batch，其余维度摊平」—— 一个非常常见的前处理
    nodes = [
        h.make_node("Shape", ["X"], ["s"]),        # s = [batch, 4]
        h.make_node("Gather", ["s", "i0"], ["d0"], axis=0),  # d0 = batch
        h.make_node("Concat", ["d0", "n1"], ["ns"], axis=0), # ns = [batch, -1]
        h.make_node("Reshape", ["X", "ns"], ["r"]),
        h.make_node("Identity", ["r"], ["Y"]),
    ]
    p = build(
        "b1_dynamic",
        nodes,
        [vi("X", ["batch", 4])],
        [vi("Y", ["batch", 4])],
        [i64([0], "i0"), i64([-1], "n1")],
    )

    got, m, ok = simp(p)
    # 直接读化简后模型的输入声明，看第 0 维是不是还是符号
    dim = m.graph.input[0].type.tensor_type.shape.dim[0]
    dim_str = dim.dim_param if dim.dim_param else dim.dim_value

    print(f"  原始 : {[n.op_type for n in onnx.load(p).graph.node]}")
    print(f"  化简 : {got}")
    print(f"  输入第 0 维 = {dim_str!r}   ← 应该是 'batch'（字符串），不是数字")
    print()

    # 关键验证：化简后还能不能吃不同 batch
    # 如果这就是 onnxsim 的正确行为，那 batch 1/3/8 都应该跑通
    onnx.save(m, "b1_dynamic_sim.onnx")
    print("  用不同 batch 跑化简后的模型:")
    for bs in (1, 3, 8):
        x = np.random.randn(bs, 4).astype(np.float32)
        y = run("b1_dynamic_sim.onnx", {"X": x})
        print(f"    batch={bs}: 跑通, max|diff|={np.abs(y - x).max():.1e}")

    print()
    print("  → 结论：默认不碰动态维。这是正确行为，因为化简不该改变模型契约。")


# ── 边界 ②：overwrite 会把动态维焊死 ────────────────────────
def boundary_overwrite():
    header("② overwrite_input_shapes：一旦指定静态，就永久焊死")

    # 同一个图，但告诉 onnxsim「把 batch 当成 1」
    got, m, ok = simp("b1_dynamic.onnx", overwrite_input_shapes={"X": [1, 4]})
    onnx.save(m, "b2_static.onnx")
    dim = m.graph.input[0].type.tensor_type.shape.dim[0]
    dim_str = dim.dim_param if dim.dim_param else dim.dim_value

    print(f"  原始 : {[n.op_type for n in onnx.load('b1_dynamic.onnx').graph.node]}")
    print(f"  overwrite 后 : {got}")
    print(f"  输入第 0 维 = {dim_str!r}   ← 从 'batch' 变成了 1")
    print()

    print("  用不同 batch 跑「被焊死」的模型:")
    for bs in (1, 3):
        x = np.random.randn(bs, 4).astype(np.float32)
        try:
            # 只关心能否跑通，不关心结果，所以不接返回值
            run("b2_static.onnx", {"X": x})
            print(f"    batch={bs}: 跑通")
        except Exception as e:
            msg = str(e).splitlines()[0]
            print(f"    batch={bs}: 失败 -> {msg[:110]}")

    print()
    print("  → 结论：这个参数是「性能 vs 通用性」的开关。")
    print("     固定 batch 的边缘部署可以加，通用服务绝对不能加。")
    print("     注意它只在「你显式传了」时才动手 —— 默认什么都不做。")


# ── 边界 ③：自定义算子只能清两端 ─────────────────────────────
def boundary_custom():
    header("③ 自定义算子：只能清两端，不能穿透")

    # 自定义算子夹在两条可化简路径中间：
    #   Identity -> MyAdd -> (Mul k) -> (Div k) -> Identity
    # 如果 MyAdd 是已知算子，Mul/Div 理论上有机会被约掉；
    # 但 onnxsim 不知道 MyAdd 干什么，所以不敢动它的上下游。
    nodes = [
        h.make_node("Identity", ["X"], ["x1"]),
        h.make_node("MyAdd", ["x1", "x1"], ["c"], domain="my_domain"),
        h.make_node("Mul", ["c", "k"], ["m1"]),
        h.make_node("Div", ["m1", "k"], ["back"]),
        h.make_node("Identity", ["back"], ["Y"]),
    ]
    p = build(
        "b3_custom",
        nodes,
        [vi("X", [1, 4])],
        [vi("Y", [1, 4])],
        [f32([5.0], "k")],
        opsets=[h.make_opsetid("", 17), h.make_opsetid("my_domain", 1)],
    )

    got, m, _ = simp(p)
    print(f"  原始   : {op_names(onnx.load(p))}")
    print(f"  化简后 : {op_names(m)}")
    print()
    print("  → 两头的 Identity 被删了（那是纯结构优化，不需要语义）。")
    print("     MyAdd 前后的 Mul/Div 一动不动（那是代数优化，需要语义）。")
    print()
    print("  为什么「不动」是对的：")
    print("    onnxsim 不知道 MyAdd 是幂等、可交换、还是有副作用。")
    print("    假设它错误地约掉 Mul/Div，而 MyAdd 实际会修改 k，结果就错了。")
    print("    所以它选择：不认识的算子 = 一堵墙。")


# ── 边界 ④：量化 QDQ 一个节点都不碰 ──────────────────────────
def boundary_qdq():
    header("④ 量化 QDQ：一个节点都不碰")

    # QDQ = QuantizeLinear -> DequantizeLinear
    # 这是 ONNX 的量化表示法：在算子前后插一对 Q/DQ 表示"这里用 int8 算"
    nodes = [
        h.make_node("QuantizeLinear", ["X", "q", "zp"], ["qx"]),
        h.make_node("DequantizeLinear", ["qx", "q", "zp"], ["dx"]),
        h.make_node("Mul", ["dx", "k"], ["m1"]),
        h.make_node("Div", ["m1", "k"], ["Y"]),
    ]
    p = build(
        "b4_qdq",
        nodes,
        [vi("X", [1, 4])],
        [vi("Y", [1, 4])],
        [f32([0.05], "q"), nh.from_array(np.array([0], np.uint8), "zp"), f32([5.0], "k")],
    )

    got, m, _ = simp(p)
    print(f"  原始   : {[n.op_type for n in onnx.load(p).graph.node]}")
    print(f"  化简后 : {got}")
    print()
    print("  → 完全一致，一个都没动。")
    print()
    print("  为什么：q (scale) 和 zp (zero_point) 是量化语义的一部分。")
    print("    Mul/Div 看起来能约掉，但 Q 是『有损』的：")
    print("      float -> int8 -> float 会丢掉精度")
    print("    约掉 QDQ 就等于把量化模型变回浮点模型，数值完全不同。")
    print()
    print("  → 量化图的化简要用专门的工具（ORT quantization tool / NNCF），")
    print("     不能指望 onnxsim。")


def main():
    boundary_dynamic()
    boundary_overwrite()
    boundary_custom()
    boundary_qdq()

    print()
    print("=" * 66)
    print("总结：四条边界背后的同一个原则")
    print("=" * 66)
    print("  onnxsim 只做「它能证明等价」的化简。")
    print()
    print("  能证明 -> 做")
    print("    Identity 删除、Cast 删除、常量折叠、Conv+BN 融合")
    print()
    print("  不能证明 -> 不做，并且这个「不做」是正确的")
    print("    动态维保持动态（改了契约就变了）")
    print("    自定义算子当墙（不知道语义）")
    print("    QDQ 不动（有损操作，约掉就变语义）")
    print("    (x*k)/k 不约（浮点下不等价）")


if __name__ == "__main__":
    main()