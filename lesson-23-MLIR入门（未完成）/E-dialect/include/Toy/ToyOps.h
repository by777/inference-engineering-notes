// Toy 方言 op 的头文件
//
// 声明所有 op 类。op 类的具体接口（getValue/getLhs 等）
// 由 mlir-tblgen 生成，这里只是"展开"它们

#ifndef TOY_TOYOPS_H
#define TOY_TOYOPS_H

// 为什么要有下面这一堆 include？（踩坑总结）
// 1. OpDefinition.h / BuiltinTypes.h / Dialect.h：
//    所有 op 类都继承 ::mlir::Op<>，需要基础定义
// 2. BytecodeOpInterface.h：带 attribute 的 op 会生成 Properties 结构，
//    自动实现 bytecode 序列化（readProperties/writeProperties）——
//    MLIR 19 的 properties 机制，缺它报 "BytecodeOpInterface is not a member"
// 3. OpImplementation.h：assemblyFormat 生成的 parse/print 用到
//    OpAsmParser/OpAsmPrinter 的完整定义，缺它报 "incomplete type"
// 4. Builders.h：生成的 build() 工厂用到 Builder/OpBuilder 完整定义
// 5. SideEffectInterfaces.h：[Pure] trait 展开后需要 MemoryEffectOpInterface

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// 关键宏：展开生成的 op 类声明
// GET_OP_CLASSES 展开后，文件里出现
//   class ConstantOp : public ::mlir::Op<...> {...};
//   class AddOp : public ::mlir::Op<...> {...};
//   class MulOp : public ::mlir::Op<...> {...};
// 都住在 mlir::toy 命名空间（来自 .td 的 cppNamespace）
#define GET_OP_CLASSES
#include "Toy/ToyOps.h.inc"

#endif // TOY_TOYOPS_H
