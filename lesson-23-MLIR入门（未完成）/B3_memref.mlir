// 实验3：memref 方言 —— 内存语义（可变缓冲区）
//
// 核心认知：memref 是"放的内存"，可读可写。类比 此前 NPU 的片上特征/权重缓冲。
// 与 tensor 的对比（本课最重要）：
//   tensor: 只读不可改（值语义），操作产生新值
//   memref: 可读可写（内存语义），store 修改原内存
module {
  func.func @main() -> i32 {
    // 分配一块 2x3 的内存（类比：在片上缓冲里划一块区域）
    %m = memref.alloc() : memref<2x3xi32>

    // 写入：把 42 写到 [0,0]（memref.store，修改内存！）
    %c42 = arith.constant 42 : i32
    %i0 = arith.constant 0 : index
    %j0 = arith.constant 0 : index
    memref.store %c42, %m[%i0, %j0] : memref<2x3xi32>

    // 读取：从 [0,0] 读出（memref.load）
    %v = memref.load %m[%i0, %j0] : memref<2x3xi32>

    // 释放内存
    memref.dealloc %m : memref<2x3xi32>

    return %v : i32   // 返回 42
  }
}