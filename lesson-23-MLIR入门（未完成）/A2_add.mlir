// 实验2：完整执行链路 7 + 5 = 12
// 先用 mlir-opt 把高级方言 lower 到 LLVM IR，再用 mlir-cpu-runner JIT 执行
func.func @main() -> i32{
    %0 = arith.constant 7 : i32
    %1 = arith.constant 5 : i32
    %2 = arith.addi %0, %1 : i32
    // arith,addi: arith方言的“整数加法操作” 类比C的 +
    // %0, %1: 两个操作数，都是i32类型
    // %2 : 结果的编号名字，类型i32
    return %2 : i32
}

// int main() {
//   int tmp0 = 7;     // %0 = arith.constant 7 : i32
//   int tmp1 = 5;     // %1 = arith.constant 5 : i32
//   int tmp2 = tmp0 + tmp1;   // %2 = arith.addi %0, %1 : i32
//   return tmp2;      // return %2 : i32
// }