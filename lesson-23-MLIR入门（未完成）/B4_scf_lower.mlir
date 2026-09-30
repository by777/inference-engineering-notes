// 实验4：scf 方言 —— 循环 + 条件（对应 C 的 for/if）
//
// 核心认知：scf.for 是结构化循环，lower 后变成汇编风格的循环骨架。
module {
  // 计算 0+1+2+3 = 6（循环累加）
  func.func @sum() -> i32 {
    %lb = arith.constant 0 : index    // 循环下界
    %ub = arith.constant 4 : index    // 循环上界（不含）
    %step = arith.constant 1 : index  // 步长
    %init = arith.constant 0 : i32    // 累加初值

    // scf.for：iter_args 是"循环携带值"（SSA 不可变，每轮产生新值传给下一轮）
    %result = scf.for %i = %lb to %ub step %step
        iter_args(%acc = %init) -> (i32) {
      // 循环体 Region：%i 当前索引，%acc 上轮累加值
      %i32 = arith.index_cast %i : index to i32 // i 从 index 转成 i32
      %new = arith.addi %acc, %i32 : i32// new = acc + i
      scf.yield %new : i32    // 新累加值传给下一轮
    }
    return %result : i32   // 0+1+2+3 = 6
  }

  // scf.if：条件分支
  func.func @check(%cond: i1) -> i32 {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %r = scf.if %cond -> i32 {
      scf.yield %c1 : i32
    } else {
      scf.yield %c2 : i32
    }
    return %r : i32
  }
}