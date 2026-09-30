// 自己的 mlir-opt：dpass-opt
// 注册 arith/func 方言 + 注册我的折叠 pass
// 这样就能跑：dpass-opt --dpass-fold-addi input.mlir

#include "Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"   // llvm::outs()

using namespace mlir;

int main(int argc, char **argv) {
  // ===== 调试辅助：打印 main 收到的命令行参数 =====
  // 想看参数就留着；不需要时删掉本段即可（不影响功能）
  llvm::outs() << "=== argc = " << argc << " ===\n";
  for (int i = 0; i < argc; ++i)
    llvm::outs() << "argv[" << i << "] = \"" << argv[i] << "\"\n";
  llvm::outs() << "==================\n";

  llvm::InitLLVM y(argc, argv);

  // 注册所有内置方言（arith/func/tensor/scf...全都有）
  DialectRegistry registry;
  registerAllDialects(registry);

  // 注册我的折叠 pass（让 --dpass-fold-addi 可用）
  dpass::registerDPassPasses();

  // 启动 mlir-opt 主循环（读文件 → 跑 pass → 输出）
  return asMainReturnCode(MlirOptMain(
      argc, argv, "dpass-opt - a toy mlir-opt with custom pass\n", registry));
}
