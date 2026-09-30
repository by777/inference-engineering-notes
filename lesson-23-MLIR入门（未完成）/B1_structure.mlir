// 实验1：MLIR 的四层结构 —— 一棵树
//
// 结构层级（从外到内）：
//   module { }                    ← 第1层：整张图（编译单元）
//     func.func { }               ← 第2层：函数（本身是 Region）
//       (函数体)                    ← 第3层：Block（语句序列）
//         scf.if { } else { }     ← 第4层：嵌套 Region（then/else 各是一个 Region）
//
// 类比：module=整个ONNX文件, func=函数, Block=顺序语句, Region=subgraph/代码块
module {
  func.func @main(%cond: i1) -> i32 {
    // 函数体 = 一个 Block（顺序执行）
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32

    // scf.if 带两个 Region：then 和 else，每个里面又是一个 Block
    %result = scf.if %cond -> i32 {
      // then Region 内的 Block
      %t = arith.addi %c1, %c1 : i32    // 1+1=2
      scf.yield %t : i32
    } else {
      // else Region 内的 Block
      %e = arith.addi %c2, %c2 : i32    // 2+2=4
      scf.yield %e : i32
    }
    return %result : i32
  }
}