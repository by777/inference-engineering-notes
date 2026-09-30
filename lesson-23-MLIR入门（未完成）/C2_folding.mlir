// 实验2：常量折叠（folding）—— pass 最经典的能力
// 编译期就把常量运算算完，省去运行时的计算
//
// 跑法：mlir-opt --canonicalize C2_folding.mlir
// 观察：所有运算（7+5、7*5、35-12、+0）编译期全算完，只剩 23
module {
  func.func @main() -> i32 {
    // 各种折叠场景
    %0 = arith.constant 7 : i32
    %1 = arith.constant 5 : i32
    %2 = arith.addi %0, %1 : i32        // 7+5 → 12（常量加折叠）
    %3 = arith.muli %0, %1 : i32        // 7*5 → 35（常量乘折叠）
    %4 = arith.subi %3, %2 : i32        // 35-12 → 23（常量减折叠）

    // 加 0 恒等：x+0 → x（数学恒等化简，不是折叠）
    %z = arith.constant 0 : i32
    %5 = arith.addi %4, %z : i32        // 23+0 → 23（+0 被删掉）

    return %5 : i32                     // 优化后只剩 constant 23
  }
}
