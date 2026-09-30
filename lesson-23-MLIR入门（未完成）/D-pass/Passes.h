// 自定义 pass 声明
#ifndef DPASS_PASSES_H
#define DPASS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace dpass {

// 我的折叠 pass：把 arith.addi 的常量折叠成结果
// 对应 C 组 canonicalize 里"折叠"功能的迷你版
std::unique_ptr<mlir::Pass> createFoldAddiPass();

// 注册所有 dpass 的 pass（dpass-opt 启动时调用）
void registerDPassPasses();

} // namespace dpass

#endif