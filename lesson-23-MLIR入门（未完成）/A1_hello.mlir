// 实验1：认识 IR 长什么样
// 最简 MLIR 模块：一个函数，返回常量 42
// .mlir 文件不是代码，是"图"的文字描述。 MLIR 在内存里维护的是一张计算图（就像你熟悉的 ONNX 图、TVM 的 Relay 图），
// .mlir 文本只是这张图被打印出来的样子。读它 = 读一张打印出来的图。

// ----------- 代码解释 -----------------
// module { ... } 这就是"一整张图"的边界。

// func.func @main() -> i32 {
//   ...
// }
// func. 是方言前缀：这个操作术语func方言。类比std:: 的std
// func：定义函数。类比C语言的函数定义
// @main： 函数的名字，@表示符号名，类比C语言的int main(...)的main
// -> i32: 返回值类型32位证书
// {...} 函数体

// %0 ： 给这个结果起的编号名字（SSA值） 类比C语言的临时变量名
// = ： 定义，不是赋值
// arith.constant: arith方言的“常量操作” 类比C的42字面量
// 42：常量的值
// :i32: 结果的类型，类比C的int

// 注意：%0 不是变量，它是一个SSA值（Single Static Assignment）静态单赋值，
// 在MLIR里每个值只能定义一次，不能被修改。%0的值就是42。
module {
    func.func @main() -> i32{
        %0 = arith.constant 42 : i32
        return %0 : i32
        // 等价C的int tmp0 = 42; return tmp0;

    }
}