// edialect-opt：能解析 toy 方言的工具
//
// 和 D 组 dpass-opt 几乎一样：注册方言 + 启动 MlirOptMain 主循环
// 区别：这里除了内置方言，还注册了我们的 toy 方言
//（MLIR 框架不认识 toy.constant/toy.add，不注册就拒绝解析）

#include "Toy/ToyDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"

using namespace mlir;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  DialectRegistry registry;
  registerAllDialects(registry);  // 注册所有内置方言（arith/scf/func...）

  // 注册我们的 toy 方言（关键一行！）
  // 之后解析器遇到 "toy.xxx" 就知道去 mlir::toy::ToyDialect 查 op
  registry.insert<mlir::toy::ToyDialect>();

  return asMainReturnCode(MlirOptMain(
      argc, argv, "edialect-opt - a toy dialect demo tool\n", registry));
}
