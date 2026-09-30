// 实验4：pass 管线组合 —— 一个 pass 同时干几件事
// 用 --pass-pipeline 指定 pass 作用范围，观察优化
//
// 跑法：mlir-opt --pass-pipeline="builtin.module(func.func(canonicalize))" C4_pipeline.mlir
// 观察：折叠(10+32=42) + 删死代码(%dead) + 合并冗余常量 + 折叠(42+42=84) 同时发生
module {
  func.func @main() -> i32 {
    // 可折叠的
    %0 = arith.constant 10 : i32
    %1 = arith.constant 32 : i32
    %2 = arith.addi %0, %1 : i32        // 10+32=42（折叠）

    // 死代码：算完没人用（DCE 删掉）
    %dead = arith.muli %0, %1 : i32     // 没人用 → 删

    // 冗余常量：42 上面已算出，这个 42 会合并（CSE）
    %3 = arith.constant 42 : i32

    %4 = arith.addi %2, %3 : i32        // 42+42=84（折叠）
    return %4 : i32                     // 优化后只剩 constant 84
  }
}
