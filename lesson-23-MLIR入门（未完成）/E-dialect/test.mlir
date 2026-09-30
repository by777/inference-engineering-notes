// 测试：用我们自定义的 toy 方言写程序
// 语法要点（来自 .td 的 assemblyFormat）：
//   - toy.constant <浮点字面量>      属性走 f64 兜底解析，不带 ": f64"
//   - toy.add %a, %b : f64           显式打印结果类型
// 语义：main 算 (2.5 + 3.5) 再乘 2.5 = 15.0
module {
  func.func @main() -> f64 {
    %0 = toy.constant 2.5
    %1 = toy.constant 3.5
    %2 = toy.add %0, %1 : f64
    %3 = toy.mul %0, %2 : f64
    return %3 : f64
  }
}
