# print_graph.py
# Lesson 09: 打印 ONNX 原始图的节点级信息（C API 没有节点级遍历接口，用 onnx python）
# 每个节点: 算子类型 / domain / 节点名 / 输入 / 输出 / 属性

import onnx

model = onnx.load("my_introspect_model.onnx")
print(f"IR version: {model.ir_version}, opset: {model.opset_import[0].version}")

for i, node in enumerate(model.graph.node):
    print(f"\nnode[{i}] {node.op_type}  (name={node.name or '(none)'})")
    print(f"  domain: {node.domain or '(default)'}")
    print(f"  inputs:  {list(node.input)}")
    print(f"  outputs: {list(node.output)}")
    for attr in node.attribute:
        # 只打印几种常见类型
        if attr.type == onnx.AttributeProto.FLOAT:
            print(f"  attr '{attr.name}' (float) = {attr.f}")
        elif attr.type == onnx.AttributeProto.INT:
            print(f"  attr '{attr.name}' (int) = {attr.i}")
        elif attr.type == onnx.AttributeProto.INTS:
            print(f"  attr '{attr.name}' (ints) = {list(attr.ints)}")
        elif attr.type == onnx.AttributeProto.STRING:
            print(f"  attr '{attr.name}' (string) = {attr.s.decode()}")
        else:
            print(f"  attr '{attr.name}' (type {attr.type})")
