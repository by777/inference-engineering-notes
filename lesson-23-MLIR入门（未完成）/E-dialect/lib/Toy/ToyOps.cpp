// Toy 方言 op 的实现
//
// op 的实现（parse/print/verify/build/接口）几乎全是生成的，
// 这里只需要"展开"生成的 .cpp.inc
//（对比：D 组手写 pass 时每个方法都要自己写；这里 tblgen 全包了）

#include "Toy/ToyOps.h"
#include "Toy/ToyDialect.h"

// 关键宏：展开生成的 op 类实现
// 同一个文件被 include 两次、由不同宏控制展开内容：
//   1. ToyDialect.cpp 里 #define GET_OP_LIST 展开 → 得到 op 类型列表
//   2. 这里 #define GET_OP_CLASSES 展开 → 得到各 op 的方法实现
#define GET_OP_CLASSES
#include "Toy/ToyOps.cpp.inc"
