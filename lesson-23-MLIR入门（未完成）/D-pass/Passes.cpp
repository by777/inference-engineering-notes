// 自定义 pass 实现：常量折叠
//
// 目标：把 %r = arith.addi(常量a, 常量b) 折叠成 %r = constant(a+b)
// 这是 C 组 canonicalize 里"折叠"功能的迷你版，你自己写一遍就懂 pass 了
//
// 总纲：这个文件在"教 MLIR 编译器一个优化规则"——
//       遍历找 addi，如果两个操作数都是常量，就换成算好的常量。
//       你写的不是普通程序，是"编译器的一个插件规则"。
//
// 三步：
//   1. runOnOperation：入口，MLIR 调它来处理 IR（相当于 pass 的主函数）
//   2. walk：遍历图里所有操作（相当于 for 循环遍历图）
//   3. match 到 addi，检查两个操作数是否常量，是则替换成结果

// ===== include：引入"工具箱" =====
// 每个头文件给你一类工具（不用管细节，知道是"MLIR 提供的零件"即可）：
#include "Passes.h"                       // 自己的声明（给别的文件看的）
#include "mlir/Dialect/Arith/IR/Arith.h"  // arith 方言（addi/constant 的定义）
#include "mlir/Dialect/Func/IR/FuncOps.h" // func 方言（函数 FuncOp 的定义）
#include "mlir/IR/Builders.h"             // OpBuilder（造新操作的"建造器"）
#include "mlir/IR/BuiltinOps.h"           // 内置操作（module 等）
#include "mlir/Pass/Pass.h"               // Pass 基类（pass 的"骨架"）
#include "mlir/Pass/PassRegistry.h"       // 注册机制（告诉 mlir-opt"有这个 pass"）
#include "mlir/Support/LLVM.h"            // LLVM 工具（StringRef 等）

using namespace mlir; // 以后写 arith::AddIOp 不用带 mlir:: 前缀（省字）

namespace dpass // 命名空间：把我们代码装进"文件夹"，防止和别人的同名类冲突
{

  // ===== 定义 pass 类 =====
  //
  // struct FoldAddiPass = 定义一个"类"（C++ 结构体里既能存数据也能存函数）
  //
  // 继承 PassWrapper<自己, 作用对象>：
  //   - ": public 基类" = 继承（单冒号是继承，双冒号 :: 是作用域）
  //     意思是 "FoldAddiPass 是一种 Pass"，自动获得 pass 的一切能力
  //     （类比：struct Dog : public Animal = "Dog 是一种 Animal"）
  //   - struct 和 class 几乎一样，唯一区别是默认访问权限：
  //       struct = public（默认公开），class = private（默认私有）
  //     MLIR 官方风格用 struct，省得写 public:
  //
  // <FoldAddiPass, OperationPass<func::FuncOp>> —— 模板参数
  // 两个参数，告诉基类"我是谁、我作用于谁"：
  //   FoldAddiPass                  自己（CRTP 技巧）     让基类知道"子类的具体类型"
  //   OperationPass<func::FuncOp>   这个 pass 作用于函数   我这个插件是"处理函数的"
  // func::FuncOp = func 方言里的"函数"操作（FuncOp 在 func 命名空间里，所以 func::FuncOp）。
  // 为什么要把自己传给基类：C++ 的 CRTP 技巧。基类需要知道"子类是谁"
  //   才能提供正确的 getOperation() 类型（返回 FuncOp 而不是通用的 Operation*）。
  struct FoldAddiPass // 定义"我的 pass 长什么样"
      : public PassWrapper<FoldAddiPass, OperationPass<func::FuncOp>>
  {
    // pass 的名字（--pass-pipeline 里用的名字，如 --dpass-fold-addi）
    // 类比：工牌上的名字。运行 --pass-pipeline="...dpass-fold-addi..." 时，
    //       MLIR 靠这个名字找到这个 pass。
    StringRef getArgument() const final { return "dpass-fold-addi"; }

    // pass 的描述（--help 里显示，类比：工牌上的职位说明）
    StringRef getDescription() const final
    {
      return "Fold arith.addi with constant operands";
    }

    // ===== 核心：runOnOperation =====
    // pass 的入口，MLIR 对"每一个函数"调用一次（相当于 pass 的主函数）
    // MLIR 说"这个函数交给你处理了"，你在这个函数里写优化逻辑
    void runOnOperation() override
    {
      // getOperation() 返回当前作用对象（这里是函数）
      // 因为你声明了 OperationPass<func::FuncOp>，所以拿到的是一个函数
      func::FuncOp func = getOperation();

      // walk：遍历函数里的所有操作（深度优先）
      // 每遇到一个 arith.addi 就调用一次 lambda（回调）
      // 类比 C：for (遍历图的所有节点) { 处理(addi); }
      // [&] = 按引用捕获外部变量：lambda 里能用外面的变量（标准写法）
      func.walk([&](arith::AddIOp add)
                {
      // ===== 检查：这个 addi 能不能折叠？ =====
      // add.getLhs() = 左操作数（一个 Value）
      //   类比：%c = arith.addi %a, %b 里，getLhs() 拿 %a，getRhs() 拿 %b
      // getDefiningOp<X>() = 问"定义这个值的操作是不是 X？"
      //   是 → 返回那个 X（非空）；不是 → 返回 null
      //   类比：问"这个变量是不是常量初始化的？"
      auto lhs = add.getLhs().getDefiningOp<arith::ConstantOp>();
      auto rhs = add.getRhs().getDefiningOp<arith::ConstantOp>();
      if (!lhs || !rhs)
        return; // 有操作数不是常量（比如变量），跳过不折叠
                // 只有两个都是常量（如 2+3）才能编译期算出来

      // ===== 取常量值并计算 =====
      // lhs.getValue() = 常量的属性值（arith.constant 5 里存着 5）
      // .dyn_cast<IntegerAttr>() = 转成整数属性（是整数才转，否则 null）
      // .getInt() = 取出整数（IntegerAttr → 5）
      auto lval = lhs.getValue().dyn_cast<IntegerAttr>().getInt();
      auto rval = rhs.getValue().dyn_cast<IntegerAttr>().getInt();
      auto result = lval + rval; // 编译期就算出来！（这就是"折叠"）

      // ===== 造新常量 + 替换 + 删除 =====
      // OpBuilder builder(add) = 在 addi 位置建一个"插入器"（建造器）
      // builder.create<arith::ConstantOp>(位置, 类型, 值) = 造一个新常量
      //   类比 C：在 addi 的位置"生成"一条新指令 ConstantOp(12)
      OpBuilder builder(add);
      auto newConst = builder.create<arith::ConstantOp>(
          add.getLoc(), add.getType(),
          builder.getIntegerAttr(add.getType(), result));

      // replaceAllUsesWith：把 addi 的所有"使用点"改成新常量
      //   比如 return %c 原本用 addi 的结果，现在改用 newConst
      //   类比：把 x = a+b 优化成 x = 5，所有用 x 的地方自动改用 5
      //  这就是手动常量折叠——自己遍历找 addi、自己算结果、
      // 用通用 API replaceAllUsesWith 把所有使用点改到新常量、再 erase 删掉旧的。replaceAllUsesWith 是通用的"改接线"工具（靠 MLIR 的 use-def 链台账），不限于折叠；"create 新的 → replace 使用 → erase 旧的"是替换任何操作的通用四步。
      add.replaceAllUsesWith(newConst.getResult());

      // erase：删掉原来的 addi（它已经没用了）
      // 顺序很重要：先 replaceAllUsesWith 再 erase！
      //   先替换所有使用，才能安全删除（否则删了 addi，用它的地方就悬空了）
      add.erase();

      // 打印日志（你能看到 pass 在干活）
      llvm::outs() << "[dpass] folded " << lval << " + " << rval << " = "
                   << result << "\n"; });
    }
  };

  // ===== 工厂函数：创建 pass 实例 =====
  // 返回 unique_ptr<Pass>（智能指针，自动管理内存，不用手动 delete）
  // 为什么需要：注册 pass 时需要一个"能创建 pass 的函数"
  std::unique_ptr<mlir::Pass> createFoldAddiPass()
  {
    return std::make_unique<FoldAddiPass>();
  }

  // ===== 注册 pass =====
  // dpass-opt 启动时调用，让 --dpass-fold-addi 可用
  // mlir::registerPass 接受一个"创建 pass 的函数"（lambda）
  void registerDPassPasses()
  {
    mlir::registerPass(
        []() -> std::unique_ptr<mlir::Pass>
        { return createFoldAddiPass(); });
  }

} // namespace dpass


// ===== 完整流程（你运行 dpass-opt 时发生了什么）=====
// 1. dpass-opt.cpp 启动 → 调用 registerDPassPasses() → 注册了 pass
// 2. MLIR 读入 test.mlir → 内存里有了 IR 图
// 3. pass manager 跑管线 → 遇到 dpass-fold-addi
// 4. 调用 runOnOperation()，对每个函数：
//    a. func.walk 遍历所有 addi
//    b. 第一个 addi：2+3，都是常量 → 折叠 → 打印 "folded 2 + 3 = 5"
//    c. 第二个 addi：5+5，都是常量 → 折叠 → 打印 "folded 5 + 5 = 10"
// 5. 输出优化后的 IR（只剩 constant 10）

// ===== 零件速查表 =====
// struct FoldAddiPass : public PassWrapper<...>   定义 pass 类（继承基类）="我是一个 Pass"
// getArgument()                                    pass 的名字 = 工牌上的名字
// runOnOperation()                                 pass 的入口 = 主函数
// getOperation()                                   当前处理的对象（函数）= 拿到函数
// func.walk(lambda)                                遍历所有操作 = for 循环遍历图
// add.getLhs().getDefiningOp<ConstantOp>()         左操作数是常量吗 = isConstant(左)
// getValue().dyn_cast<IntegerAttr>().getInt()      取常量的整数值 = 取出数值
// OpBuilder + create                               造新常量 = 生成新指令
// replaceAllUsesWith                               使用点改用新值 = 替换引用
// erase()                                          删掉旧操作 = 删除指令
// createFoldAddiPass()                             工厂 = 造 pass 的流水线
// registerDPassPasses()                            注册 = 告诉 MLIR"有这号 pass"
