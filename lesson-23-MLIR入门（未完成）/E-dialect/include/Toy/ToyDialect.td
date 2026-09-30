// Toy 方言的定义（.td = TableGen 源文件）
//
// .td 是 MLIR 的"方言/op 说明书"：
//   mlir-tblgen 读它 → 生成 C++ 代码（.inc 文件）
// 所以真正的工作是写 .td，手写 C++ 只是"展开"生成的代码
// .td 是什么：
// TableGen 是 LLVM 的一套"数据描述语言"。
// MLIR 用它写方言/op 的声明，mlir-tblgen（LLVM 的代码生成器）读 .td 产出 .inc C++ 代码。你要写的绝大部分"声明"都在 .td 里，C++ 文件只是薄薄一层展开。 这就是 MLIR 和手写 C++ 工程最大的不同。
#ifndef TOY_TOYDIALECT_TD
#define TOY_TOYDIALECT_TD

// 引入 MLIR 的 TableGen 基础定义（Op/Dialect/type/attr 等都在里面）
// 注意：include 路径相对 mlir-tblgen 的 -I 参数（后面 CMake 里配）
include "mlir/IR/OpBase.td"

// ===== 方言声明 =====
// def 定义一个"记录"(record)，Toy_Dialect 是记录名
// : Dialect 表示继承自 TableGen 里的 Dialect 类
// mlir-tblgen -gen-dialect-decls 读它 → 生成 C++ 的 ToyDialect 类
//   类名规则：def 名去掉尾部 "_Dialect"：Toy_Dialect → ToyDialect
def Toy_Dialect : Dialect {
  // name：IR 文本里 op 的前缀。比如 toy.constant 的 "toy"
  let name = "toy";

  // summary/description：文档用，可省略
  let summary = "A minimal dialect for the E-dialect demo";
  let description = [{
    The `toy` dialect is the minimal example dialect used to learn
    how to define a custom dialect with TableGen + mlir-tblgen.
  }];

  // cppNamespace：生成的 ToyDialect 类放在哪个 C++ 命名空间
  // 惯例是 ::mlir::<方言名>，我们写 mlir::toy
  let cppNamespace = "::mlir::toy";
}

#endif // TOY_TOYDIALECT_TD