// 实验2：tensor 方言 —— 值语义（不可变张量）
//
// 核心认知：tensor 是"算的值"，不可变。对 tensor 的任何操作都产生新 tensor，
// 原 tensor 不变。类比 ONNX 图里流动的 tensor。
module {
  func.func @main() -> tensor<2x3xf32> {
    // tensor 常量：dense 表示完整数据
    %t = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>

    // 读元素：索引必须是 SSA 值（index 类型）！
    // （坑：不能写 tensor.extract %t[0, 1]，静态索引不行）
    %i = arith.constant 0 : index
    %j = arith.constant 1 : index
    // : tensor<2x3xf32> 是操作数 %t 的类型，不是结果 %e 的类型。
    // 语法把操作数类型写在了操作数后面，省略了结果类型。
    %e = tensor.extract %t[%i, %j] : tensor<2x3xf32>

    // 生成新 tensor：每个元素 +1，产生新值，原 tensor 不变！
    %c1 = arith.constant 1.0 : f32
    // tensor.generate = "按位置造一个新 tensor"。
    // tensor<2x3xf32> 是结果类型：造出来的是个 2x3 的 float tensor
    %new = tensor.generate {
      // %a/%b 是自动传入的"当前位置"。
      ^bb0(%a: index, %b: index): // ^bb0(%a: index, %b: index): —— 规则入口，接收位置
        %val = tensor.extract %t[%a, %b] : tensor<2x3xf32>
        %sum = arith.addf %val, %c1 : f32
        tensor.yield %sum : f32
    } : tensor<2x3xf32>

    return %new : tensor<2x3xf32>
  }
}