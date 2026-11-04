"""实验 1：最小化简 —— 验证 onnxsim 的三类动作

目的
    用一个人造的 8 节点图，一次性看清 onnxsim 在做什么，
    并区分「消除」「常量折叠」「融合」这三种不同的动作。

    这个图不是随意拼的，每个节点都是故意放的诱饵：
      Identity / Cast    -> 测「无操作算子消除」
      Mul(c1,c2)         -> 测「常量折叠」（两个常量乘出一个新常量）
      MatMul + Add       -> 测「算子融合」（两个变成一个 Gemm）
      Mul(a1,k)+Div(k)   -> 测「它什么时候选择不动手」
                            浮点下 (x*k)/k != x，理论上不该约掉

预期结论
    8 节点 -> 3 节点，且产物仍是 ai.onnx 标准算子（不是 com.microsoft）

运行
    python3 e1.py
"""

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh


def C(v, n):
    """造一个 float32 常量（initializer）。"""
    return nh.from_array(np.asarray(v, np.float32), n)


# ── 1. 造图 ──────────────────────────────────────────────────
# 权重和 bias 是「真正的」模型参数，化简后应该还在
W = C(np.random.randn(4, 4) * 0.1, "W")
b = C(np.random.randn(4) * 0.1, "b")
# c1/c2 是「假参数」，唯一用途是被折叠掉：
# 2.0 * 3.0 -> 6.0，两个 initializer 应该合并成一个
c1 = C([2.0], "c1")
c2 = C([3.0], "c2")

nodes = [
    # ① 无操作：Identity 删掉后语义完全不变
    h.make_node("Identity", ["X"], ["x1"]),
    # ② 无操作：float -> float 的 Cast 没有信息损失
    h.make_node("Cast", ["x1"], ["x2"], to=T.FLOAT),
    # ③ 测融合：MatMul + Add 应该被合成一个 Gemm
    h.make_node("MatMul", ["x2", "W"], ["mm"]),
    h.make_node("Add", ["mm", "b"], ["a1"]),
    # ④ 测常量折叠：两个常量相乘
    h.make_node("Mul", ["c1", "c2"], ["k"]),
    # ⑤ 测「克制的化简」：(x*k)/k 不该被约成 x —— 浮点下不等价
    h.make_node("Mul", ["a1", "k"], ["scaled"]),
    h.make_node("Div", ["scaled", "k"], ["back"]),
    # ⑥ 又一个无操作
    h.make_node("Identity", ["back"], ["Y"]),
]

g = h.make_graph(
    nodes,
    "redundant",
    [h.make_tensor_value_info("X", T.FLOAT, [1, 4])],
    [h.make_tensor_value_info("Y", T.FLOAT, [1, 4])],
    [W, b, c1, c2],
)
# ir_version=10 是 ONNX 1.16+ 的版本；太高会让老版本 ORT 打不开
m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)])
m.ir_version = 10
onnx.save(m, "redundant.onnx")

print("原始 :", [n.op_type for n in m.graph.node], len(m.graph.initializer), "inits")

# ── 2. 化简 ──────────────────────────────────────────────────
# 坑：simplify() 返回的是 (model, ok) 元组，不是 model。
# 写成 simp = onnxsim.simplify(m) 再 simp.graph 会 AttributeError。
r = onnxsim.simplify(m)
print("返回 :", type(r).__name__, r if not isinstance(r, tuple) else len(r))
simp, ok = r

onnx.save(simp, "redundant_sim.onnx")
print(
    "简化 :", [n.op_type for n in simp.graph.node], len(simp.graph.initializer), "inits"
)
# ok 默认只做结构校验（onnx.checker），不验数值。
# 要验数值得传 check_n=3（见实验 5）。
print("ok   :", ok)

# 确认常量折叠真的发生了：k 应该是 6.0，而不是还在图里算
print(
    "k 的值:",
    [
        onnx.numpy_helper.to_array(i).tolist()
        for i in simp.graph.initializer
        if i.name == "k"
    ],
)

# ── 3. 数值对拍 ──────────────────────────────────────────────
# 化简的前提是「等价」，所以必须比数值，不能只看节点数变少了。
x = np.random.randn(1, 4).astype(np.float32)
run = lambda p: ort.InferenceSession(p, providers=["CPUExecutionProvider"]).run(
    None, {"X": x}
)[0]

a, c = run("redundant.onnx"), run("redundant_sim.onnx")
# bit-equal 比 max|diff| 更严格：浮点重排可能让 diff=0 但 bit 不同
print(
    "max|diff|:",
    np.abs(a - c).max(),
    " bit-equal:",
    np.array_equal(a.view(np.int32), c.view(np.int32)),
)
