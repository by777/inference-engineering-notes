// 实验3：死代码消除（DCE）—— 没用的计算被删掉
// 算出来但没人用的值，编译器直接删掉，省去运行时的浪费
//
// 怎么判断死活：从 return 反向追踪。
//   return %2 需要 %2 → %2 活的；%2 需要 %0/%1 → 它们活的
//   %dead1/%dead2/%dead3 没被任何活值用 → 死的 → 删
//   关键：依赖死代码的也是死的（dead3 依赖 dead1/dead2，跟着删）
//
// 跑法：mlir-opt --canonicalize C3_dce.mlir
// 观察：三个死代码全被删，只剩有用的 3
module {
  func.func @main() -> i32 {
    // 有用的值（会被 return 用到 → 活的）
    %0 = arith.constant 1 : i32
    %1 = arith.constant 2 : i32
    %2 = arith.addi %0, %1 : i32        // 1+2=3，会被 return 用

    // 死代码：算出来但没人用！（→ 死的，被删）
    %dead1 = arith.muli %0, %1 : i32    // 1*2=2，没人用
    %dead2 = arith.subi %1, %0 : i32    // 2-1=1，没人用
    %dead3 = arith.addi %dead1, %dead2 : i32  // 依赖死代码，也是死的

    return %2 : i32                     // 只返回 3（优化后只剩它）
  }
}
