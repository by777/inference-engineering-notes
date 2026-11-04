"""实验 2：迭代到不动点 —— 为什么 onnxsim 要跑不止一遍

目的
    实验 1 的图一遍就能化简完。这里造一个「一遍不够」的图，
    用实测数据说明 onnxsim 内部是「交替执行 + 比对模型指纹 + 跑到不变为止」。

为什么一遍不够
    见下面 build() 里的注释：每一层 BN 的 scale 都是上一轮算出来的，
    fuse_bn_into_conv 要求参数已是 initializer，所以必须等上一轮折叠完成。

怎么控制迭代次数
    没有 Python 参数，只有环境变量 ONNXSIM_FIXED_POINT_ITERS（默认 50）。
    它每次调用 simplify() 都会重新读，所以在同一进程里循环改 os.environ 是有效的。

一个必须知道的坑
    收敛警告是 C++ 层直接写 fd 1 的，
    Python 的 redirect_stdout / redirect_stderr 都抓不到 ——
    只能用 subprocess 看子进程的原始输出。所以本脚本用子进程跑每一组。

运行
    python3 e2.py
"""

import io
import contextlib
import json
import os
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

PATH = "deep_bn.onnx"
SIMPATH = "deep_bn_sim.onnx"


def C(v, n):
    """造一个 float32 常量（initializer）。"""
    return nh.from_array(np.asarray(v, np.float32), n)


def vi(name, shape):
    return h.make_tensor_value_info(name, T.FLOAT, shape)


def build(depth=3):
    """造 depth 层 BN 串行的图，每层 scale 都由上一条 Mul 算出。"""
    Wc = C(np.random.randn(8, 3, 3, 3) * 0.1, "Wc")
    inits = [
        Wc,
        C(np.ones(8) * 2.0, "a"),
        C(np.ones(8) * 0.5, "b"),
        C(np.zeros(8), "beta"),
        C(np.zeros(8), "mean"),
        C(np.ones(8), "var"),
    ]
    nodes = [
        h.make_node(
            "Conv", ["X", "Wc"], ["cv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        ),
        # ① sc1 是「算出来」的，不是 initializer —— BN 融不了，除非先折叠
        h.make_node("Mul", ["a", "b"], ["sc1"]),
        # ② 这个 BN 的参数 sc1 还没折好，第 1 轮理论上融不掉
        h.make_node("BatchNormalization", ["cv", "sc1", "beta", "mean", "var"], ["b1"]),
    ]
    prev = "b1"
    for i in range(2, depth + 1):
        # ③ 每多一层，就多一个「依赖上一轮折叠结果」的 scale
        nodes.append(h.make_node("Mul", [f"sc{i-1}", f"c{i}"], [f"sc{i}"]))
        nodes.append(
            h.make_node(
                "BatchNormalization",
                [prev, f"sc{i}", "beta", "mean", "var"],
                [f"b{i}"],
            )
        )
        inits.append(C(np.ones(8) * 0.5, f"c{i}"))
        prev = f"b{i}"
    # ④ 最后一个 BN 的输出直接作为图输出（避免多一个 Identity）
    nodes[-1].output[0] = "Y"

    g = h.make_graph(
        nodes, "deep_bn", [vi("X", [1, 3, 8, 8])], [vi("Y", [1, 8, 8, 8])], inits
    )
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 15)])
    m.ir_version = 10
    onnx.save(m, PATH)
    return m


# 子进程里跑的代码：必须在新进程里，因为环境变量在 import 之后不一定生效
CHILD = r"""
import io, contextlib, json, sys
import onnx, onnxsim

# onnxsim 的统计表格走 Python 的 stdout，可以吞掉
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    model, ok = onnxsim.simplify(onnx.load(sys.argv[1]))

# 用标记符，方便父进程从混杂输出里挑出结果
print("@@@" + json.dumps({
    "nodes": [n.op_type for n in model.graph.node],
    "n_init": len(model.graph.initializer),
}))
"""


def run_with_iters(path, iters):
    """在新进程里跑一次 simplify，把迭代上限固定为 iters。"""
    env = dict(os.environ, ONNXSIM_FIXED_POINT_ITERS=str(iters))
    p = subprocess.run(
        [sys.executable, "-c", CHILD, path],
        capture_output=True,
        text=True,
        env=env,
    )
    line = [l for l in p.stdout.splitlines() if l.startswith("@@@")]
    if not line:
        return {"nodes": ["<失败>"], "n_init": -1, "converged": False}
    r = json.loads(line[-1][3:])
    # 警告原文：「WARNING: the simplification stopped because of timeout.」
    # 注意它出现在 stdout 里，不是 stderr
    r["converged"] = "WARNING" not in p.stdout
    return r


def quiet_simplify(model):
    """同进程调用时，吞掉统计表格（但 C++ 的 WARNING 吞不掉，会漏出来）。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return onnxsim.simplify(model)


def main():
    m = build(3)
    print(
        "原始 :", [n.op_type for n in m.graph.node], len(m.graph.initializer), "inits"
    )
    print()

    # ── 明线：改变迭代上限，看结果什么时候稳定 ──
    print(f"{'iters':>5}  {'节点':<46} {'inits':>5}  {'已收敛':>6}")
    print("-" * 72)
    for it in (1, 2, 3, 4, 5, 10):
        r = run_with_iters(PATH, it)
        flag = "是" if r["converged"] else "否"
        print(f"{it:>5}  {str(r['nodes']):<46} {r['n_init']:>5}  {flag:>6}")
    print()

    # ── 暗线：手动多跑几遍，是否等效于内部自动迭代 ──
    # 这一段会漏出几行 WARNING，是正常的（C++ 直接写 fd 1）
    print("手动循环调用 simplify（每次只给 1 轮迭代）:")
    os.environ["ONNXSIM_FIXED_POINT_ITERS"] = "1"
    mm = onnx.load(PATH)
    for k in range(1, 5):
        mm, _ = quiet_simplify(mm)
        print(f"  第 {k} 次: {[n.op_type for n in mm.graph.node]}")
    os.environ.pop("ONNXSIM_FIXED_POINT_ITERS", None)
    print()

    # ── 数值校验：化简到最后是否仍然等价 ──
    os.environ["ONNXSIM_FIXED_POINT_ITERS"] = "10"
    simp, _ = quiet_simplify(onnx.load(PATH))
    os.environ.pop("ONNXSIM_FIXED_POINT_ITERS", None)
    onnx.save(simp, SIMPATH)

    x = np.random.randn(1, 3, 8, 8).astype(np.float32)

    def run(p):
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        return s.run(None, {"X": x})[0]

    a, b = run(PATH), run(SIMPATH)
    print(
        f"数值: max|diff|={np.abs(a-b).max():.3e}  "
        f"bit-equal={np.array_equal(a.view(np.int32), b.view(np.int32))}"
    )


if __name__ == "__main__":
    main()
