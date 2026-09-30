// Toy 方言的 op 定义
//
// 每个 op 一个 def 块：
//   声明输入(arguments)/输出(results)/文本格式(assemblyFormat)
// mlir-tblgen -gen-op-decls/-gen-op-defs 读它 → 生成 op 的 C++ 类
// ============================================================
// TableGen 整个语言其实只有 4 种语法
// 别看 .td 文件花里胡哨，全部语法就这些：

// 语法	含义	C++ 类比
// def 名字 : 模板<参数> { let 字段 = 值; }	定义一个具体记录	声明一个对象
// class 名字<参数> : 模板<参数> { ... }	定义一个可复用模板	template<typename T>
// let 字段 = 值;	给字段赋值	成员变量赋值
// include "xxx.td"	引入其他定义	#include
// ============================================================

#ifndef TOY_TOYOPS_TD
#define TOY_TOYOPS_TD

// 先引入方言定义（Toy_Dialect 在这里）
include "Toy/ToyDialect.td"
// 再引入 Op 基类（其实 ToyDialect.td 里已带 OpBase.td，这里显式写出
// 是为了让读者看清依赖，注释掉这行也能编过）
include "mlir/IR/OpBase.td"
// 引入 Pure trait 定义（[Pure] 里那个 Pure 在这定义！）
// OpBase.td 的 include 链里没有它——官方 standalone 也显式 include 这行
// 踩坑：缺这行报 "error: Variable not defined: 'Pure'"
include "mlir/Interfaces/SideEffectInterfaces.td"

// ===== 自定义 op 基类 =====
// 我们不直接写 Op<Toy_Dialect, ...>，而是先定义一个 Toy_Op 子类
// 好处：以后加 op 只要写 Toy_Op<"名字", [trait...]>，不用重复写方言参数
// class 定义：带参数的可复用模板（TableGen 的 class 类似 C++ 模板）
//   <string mnemonic, list<Trait> traits = []>
//     mnemonic：op 短名（"constant"/"add"/"mul"），拼出 toy.xxx
//     traits：op 特性列表，默认空
class Toy_Op<string mnemonic, list<Trait> traits = []>
    : Op<Toy_Dialect, mnemonic, traits>;

// ===== toy.constant =====
// 常量：无输入，一个 f64 输出，带一个 f64 属性存常量值
// 对应 IR 写法：%0 = toy.constant 2.5
// [Pure] 是 TableGen 的列表字面量（列表只有这一个元素）：[Pure] 
// 里那个 Pure，是 MLIR **预定义的一个 trait（特性）**的名字。
// 写成 [Pure, ...] 就是给 op 贴多个特性标签。
// 标签	含义	C++ 里叫
// 无副作用	不写内存、不打日志、不产生外部可观察效果	MemoryEffectOpInterface（无 MemoryEffects::Write）
// 确定性可折叠	输入相同 → 输出一定相同，可在编译期算掉	ConditionallySpeculatable + 恒可投机
// 对应到你的工程常识：

// 加法 2.5 + 3.5：谁都能算，算一万遍结果一样，不碰任何内存 → 纯函数
// 反例：printf、写文件、读硬件寄存器 → 有副作用，不能标 Pure
// 编译器对 Pure op 可以合法做这些优化——这也是为什么标 Pure 有实际意义：

// 优化	例子	不标 Pure 会怎样
// 常量折叠 (folder)	编译期直接算 toy.add(2.5, 3.5) → 换成 6.0	只能运行时算
// 公共子表达式消除 (CSE)	toy.add %0, %1 出现两次 → 删一个，共用结果	不敢删——万一副作用执行两次有影响呢
// 死代码消除 (DCE)	结果没被用 → 整个 op 删掉	不敢删——万一它有副作用呢
// 投机执行 (speculation)	提前算、算错就丢弃（不影响正确性）	不能
def Toy_ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "Constant scalar";

  // arguments (ins ...)：输入。这里没有 SSA 操作数，只有一个属性
  //   F64Attr:$value  = 一个 f64 属性，绑定名 value
  //   (属性的"值"编译期就定死，写在 IR 的属性字典里)
  let arguments = (ins F64Attr:$value);

  // results (outs ...)：输出
  //   F64:$result = 一个 f64 类型结果，绑定名 result
  let results = (outs F64:$result);

  // assemblyFormat：IR 文本长什么样（自定义语法，不用写 parse/print）
  //   $value      → 打印属性值
  //   attr-dict   → 其余属性（如果有）原样放在这（必须写，规范要求）
  //   生成效果：toy.constant 2.5
  let assemblyFormat = "$value attr-dict";
}

// ===== toy.add =====
// 两个 f64 输入，一个 f64 输出
// 对应 IR 写法：%2 = toy.add %0, %1 : f64
def Toy_AddOp : Toy_Op<"add", [Pure]> {
  let summary = "Add two f64 values";
  let description = [{
    The `toy.add` op adds two f64 values.
  }];

  let arguments = (ins F64:$lhs, F64:$rhs);
  let results = (outs F64:$result);

  // 格式：$lhs `,` $rhs attr-dict `:` type($result)
  //   type($result) = 打印结果类型。SSA 操作数(operaand)的类型
  //   通常不写（可以从 def 推断），所以这里只有 : f64 一个类型
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// ===== toy.mul =====
// 两个 f64 输入，一个 f64 输出（和 add 同构，展示"一个方言多个 op"）
// 对应 IR 写法：%2 = toy.mul %0, %1 : f64
def Toy_MulOp : Toy_Op<"mul", [Pure]> {
  let summary = "Multiply two f64 values";
  let description = [{
    The `toy.mul` op multiplies two f64 values.
  }];

  let arguments = (ins F64:$lhs, F64:$rhs);
  let results = (outs F64:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

#endif // TOY_TOYOPS_TD