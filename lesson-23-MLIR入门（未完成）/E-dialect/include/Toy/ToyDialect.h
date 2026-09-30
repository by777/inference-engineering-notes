// Toy 方言的头文件
//
// 手写的部分很少：include 生成的代码 + 声明需要的类
// 大多数代码（方言类本体）都是 mlir-tblgen 生成的

#ifndef TOY_TOYDIALECT_H
#define TOY_TOYDIALECT_H

#include "mlir/IR/Dialect.h"

// include mlir-tblgen 生成的方言类声明
// 这个 .inc 文件不存在于源码里！是 CMake 在编译前用 mlir-tblgen
// 从 ToyDialect.td 生成的（生成到 build/gen/Toy/，靠 -I 找到）
#include "Toy/ToyDialect.h.inc"

#endif // TOY_TOYDIALECT_H
