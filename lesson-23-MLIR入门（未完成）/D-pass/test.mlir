// 测试输入：两个 addi 都是常量操作数，都会被折叠
module {
  func.func @main() -> i32 {
    %a = arith.constant 2 : i32
    %b = arith.constant 3 : i32
    %c = arith.addi %a, %b : i32       // 2+3，会被折叠成 5
    %d = arith.addi %c, %c : i32       // 5+5，也会被折叠成 10（%c 已是常量）
    return %d : i32
  }
}
