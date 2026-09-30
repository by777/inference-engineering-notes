// 实验4b：公共子表达式消除（CSE）+ canonicalize
// 两个 pass 串起来，用 --mlir-print-ir-after-all 看每步变化
//
// 跑法：mlir-opt --pass-pipeline="builtin.module(func.func(cse,canonicalize))" \
//        --mlir-print-ir-after-all C4b_cse.mlir
// 观察两步变化：
//   第1步 CSE 后：两个 3*4 合并成一个 %0（去重）
//   第2步 canonicalize 后：3*4=12, 12+12=24（折叠）
// 整个链路：两个 3*4 → CSE 合并 → 折叠 → 24
module {
  func.func @main() -> i32 {
    %0 = arith.constant 3 : i32
    %1 = arith.constant 4 : i32
    // 两个完全相同的计算：3*4
    %a = arith.muli %0, %1 : i32   // 3*4
    %b = arith.muli %0, %1 : i32   // 3*4（重复！CSE 会合并成同一个 %0）
    %c = arith.addi %a, %b : i32   // (3*4)+(3*4)，优化后 %a/%b 共用 %0
    return %c : i32                // 最终折叠成 24
  }
}
