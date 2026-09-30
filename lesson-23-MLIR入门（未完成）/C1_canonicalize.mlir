// 实验1：canonicalize 热身 —— 看 pass 对 IR 做了什么
//
// canonicalize 是一组小优化的集合：
//   1. 折叠：常量运算编译期算完（2+3 → 5）
//   2. 化简：恒等化简（x+0 → x）
//   3. 规范化：等价写法统一成同一种
//
// 跑法：mlir-opt --canonicalize C1_canonicalize.mlir
// 观察：优化前 vs 优化后对比（addi 全消失，只剩常量 10）
module {
  func.func @main() -> i32 {
    // 常量折叠：2+3 编译期就算成 5
    %0 = arith.constant 2 : i32
    %1 = arith.constant 3 : i32
    %2 = arith.addi %0, %1 : i32     // 2+3 → 5（折叠）

    // 冗余常量：这个 5 和上面算出来的 5 相同，会合并
    %3 = arith.constant 5 : i32

    // 结果：(2+3)+5 = 5+5 = 10
    %4 = arith.addi %2, %3 : i32     // 5+5 → 10（折叠）
    return %4 : i32                  // 优化后整个函数只剩 constant 10
  }
}
