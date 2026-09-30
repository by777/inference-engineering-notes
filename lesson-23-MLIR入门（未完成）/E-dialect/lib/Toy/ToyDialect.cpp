// Toy 方言的实现
//
// 手写部分只有 initialize()：把 op 注册进方言
//（相当于告诉 MLIR "toy 方言有哪些 op"）

#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

using namespace mlir;
using namespace mlir::toy;

// include 生成的方言类实现（构造/析构等）
#include "Toy/ToyDialect.cpp.inc"

// initialize：方言被加载时调用，注册所有 op
// addOperations<...> 的列表来自生成的 GET_OP_LIST：
//   tblgen 遍历 .td 里所有 def Toy_*Op，生成一个
//   #define GET_OP_LIST 展开的 op 类型列表
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Toy/ToyOps.cpp.inc"
      >();
}
