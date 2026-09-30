# Lesson 23-E：自定义方言（工程全解）

## 〇、本课定位（先读）

### E 组在整节课里的位置

| | D 组 | E 组 |
|---|---|---|
| 造什么 | 造 **pass**（变换） | 造 **方言**（词表） |
| 比喻 | 给工厂加一个**工位** | 给工厂加一套**新零件标准** |
| 产物 | `dpass-opt`（能优化 IR） | `edialect-opt`（能认识 `toy.xxx`） |
| 对应你的经验 | 此前 NPU 的"优化脚本" | 此前 NPU 的"算子宏/指令定义" |

**一句话**：E 组是**教 MLIR 认识一门新语言**。

### 为什么需要 E 组

MLIR 框架**天生不认识** `toy.constant` 这种东西：

```bash
$ mlir-opt test.mlir
test.mlir:3:21: error: operation being parsed with an unregistered dialect.
    %0 = "toy.constant"() : () -> f64
```

**这个报错就是"为什么需要 E 组"的最好证明**——MLIR 严格，不认识就拒绝。要加一门新方言，必须做三件事：

1. **告诉 MLIR 这门方言叫什么** → 注册方言
2. **告诉它有哪些 op、长什么样** → 用 `.td` 定义
3. **给它一个认识这门方言的工具** → 自建 `edialect-opt`

### 最重要的一句话（全文核心）

> **你写"说明书"（`.td`），`mlir-tblgen` 写"代码"（`.inc`），你再用薄薄的"插头"（`.h`/`.cpp`）把代码接进工具。**

**这和你在 此前 NPU 写算子的模式完全同构**：

| 此前 NPU | MLIR E 组 |
|---|---|
| 写宏 / 脚本描述 | 写 `.td` 说明 |
| 脚本工具翻译 | `mlir-tblgen` 翻译 |
| 生成 asm | 生成 `.inc`（C++ 代码） |
| 胶水/调用代码 | 手写 `.h`/`.cpp` 插头 |

你早就在用"描述驱动 + 代码生成"这个模式了，MLIR 只是把它标准化成了一套工具链。

---

## 一、工程结构

```
E-dialect/
├── include/Toy/
│   ├── ToyDialect.td     ← 【你写】方言声明（源头）
│   ├── ToyOps.td         ← 【你写】3 个 op 声明（源头）
│   ├── ToyDialect.h      ← 【你写】插头：include 生成的方言类
│   └── ToyOps.h          ← 【你写】插头：include 生成的 op 类
├── lib/Toy/
│   ├── ToyDialect.cpp    ← 【你写】插头：initialize() 注册 op
│   └── ToyOps.cpp        ← 【你写】插头：展开生成的 op 实现
├── tools/
│   └── edialect-opt.cpp  ← 【你写】工具入口（相当于 D 组的 dpass-opt）
├── build/gen/Toy/        ← 【工具生成】4 个 .inc（1117 行！）
│   ├── ToyDialect.h.inc
│   ├── ToyDialect.cpp.inc
│   ├── ToyOps.h.inc
│   └── ToyOps.cpp.inc
├── CMakeLists.txt        ← 【你写】串起"先生成、再编译"
└── test.mlir             ← 【你写】用 toy 方言写的测试程序
```

### 分工全景图

```
【源头】你写"说明书"          【插头】你写"胶水"        【产物】工具生成
┌─────────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ ToyDialect.td       │─┐    │ ToyDialect.h     │◄─────│ ToyDialect.h.inc │
│  (方言叫什么)        │ │    │ ToyOps.h         │◄─────│ ToyOps.h.inc     │
│                     │ │    │                  │      │                  │
│ ToyOps.td           │─┤    │ ToyDialect.cpp   │◄─────│ ToyDialect.cpp.inc│
│  (有哪些 op)         │ │    │ ToyOps.cpp       │◄─────│ ToyOps.cpp.inc   │
└─────────────────────┘ │    └──────────────────┘      └──────────────────┘
        │               │                                  ▲
        └───────────────┴──────────────────────────────────┘
                  mlir-tblgen 读 .td → 生成 .inc
```

### 工作量对比（最能说明问题）

| 类别 | 行数 |
|---|---|
| 你手写的 5 个 C++ 文件 | **约 120 行** |
| 工具生成的 4 个 `.inc` | **1117 行** |

> 真正的"重活"（parse/print/verify/build/序列化）全是生成的。**你写的是"声明"，不是"实现"。**

---

## 二、TableGen 语法（只有 4 条）

`.td` 文件看着吓人，语法**一共就 4 条**：

| 语法 | 含义 | C++ 类比 |
|---|---|---|
| `include "xxx.td"` | 引入别的定义 | `#include` |
| `class 名字<参数> : 父类 { ... }` | 定义**可复用模板** | `template<typename T>` |
| `def 名字 : 模板<参数> { ... }` | 用模板造一个**具体东西** | 声明一个对象 |
| `let 字段 = 值;` | 给字段赋值 | 成员变量赋值 |

**就这些。** 别的 `Op`、`Dialect`、`F64Attr`、`Pure` **都不是 TableGen 语法**，是 MLIR 预先准备好的"素材库"（当标准库用）。

### 2.1 尖括号是干嘛的

```tablegen
class Toy_Op<string mnemonic, list<Trait> traits = []>
//            ↑类型     ↑参数名    ↑类型      ↑默认值
```

TableGen 传参**一律用尖括号**，且**类型和值混着传**：

| 传的东西 | 例子 | 类型 |
|---|---|---|
| 字符串 | `"constant"` | `string` |
| 列表 | `[Pure]` | `list<Trait>` |
| 类型 | `F64` | TableGen 内置类型 |

`traits = []` 是**默认参数**——所以 `Toy_Op<"add">` 不写第二个参数也行。

> ⚠️ `.td` **不是 C++**，g++ 不编译它，是 `mlir-tblgen` 读它。
> 它长得像 C++ 模板纯粹是**故意的**（让你眼熟）。

### 2.2 ⚠️ include 是显式的（重要规律）

TableGen 的 `include` **没有 C++ 头文件那样的传递性保证**。用到什么符号，就必须 include 定义它的 `.td`。

**案例**：`[Pure]` 报错
```
error: Variable not defined: 'Pure'
```
原因：`Pure` 定义在 `mlir/Interfaces/SideEffectInterfaces.td`，而 `mlir/IR/OpBase.td` 的 include 链**不带它**。

解法：
```tablegen
include "mlir/Interfaces/SideEffectInterfaces.td"
```
（官方 `mlir/examples/standalone/` 也显式 include 这行。）

---

## 三、`ToyDialect.td` 逐行讲（方言声明）

```tablegen
#ifndef TOY_TOYDIALECT_TD
#define TOY_TOYDIALECT_TD

// ① 引入素材库（Dialect/Op/F64 等基础定义都在这）
include "mlir/IR/OpBase.td"

// ② 定义一个方言，记录名叫 Toy_Dialect，继承 Dialect
def Toy_Dialect : Dialect {
  // ③ IR 里的前缀：toy.xxx
  let name = "toy";

  // ④ 文档用（可省略）
  let summary = "A minimal dialect for the E-dialect demo";

  // ⑤ 生成的 C++ 类放哪个命名空间
  let cppNamespace = "::mlir::toy";
}

#endif
```

### 命名规则（坑）

`def Toy_Dialect` 生成 C++ 类时会**自动去掉尾部的 `_Dialect`**：

```
Toy_Dialect  →  类名 ToyDialect
```

所以 C++ 里写 `mlir::toy::ToyDialect`。

### 这份文件生成 2 个产物

```bash
mlir-tblgen -gen-dialect-decls  ToyDialect.td  # → ToyDialect.h.inc（类声明）
mlir-tblgen -gen-dialect-defs   ToyDialect.td  # → ToyDialect.cpp.inc（构造/析构）
```

---

## 四、`ToyOps.td` 逐行讲（op 定义）

### 4.1 先定义"基类"省事

```tablegen
class Toy_Op<string mnemonic, list<Trait> traits = []>
    : Op<Toy_Dialect, mnemonic, traits>;
```

**为什么要这行？** 每个 op 真正要写的是 `Op<Toy_Dialect, "名字", [traits]>`。把固定的 `Toy_Dialect` 藏进 `Toy_Op`，以后写 op 只要：

```tablegen
def Toy_AddOp : Toy_Op<"add", [Pure]> { ... }   // 只写变化的部分
```

### 4.2 一个完整 op 的解剖

```tablegen
def Toy_AddOp : Toy_Op<"add", [Pure]> {          // 名字 = toy.add，带 Pure 特性
  let summary = "Add two f64 values";            // 文档（可省略）

  let arguments = (ins F64:$lhs, F64:$rhs);      // 输入：两个 f64 操作数
  let results   = (outs F64:$result);            // 输出：一个 f64

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

### 4.3 `(ins ...)` 里为什么分两类

op 的输入分两类，**必须区分**：

| 类别 | 写法 | 例子 | 什么时候定值 |
|---|---|---|---|
| **属性 Attr** | `F64Attr:$value` | `toy.constant 2.5` 的 `2.5` | **编译期**（写死在 IR 里） |
| **操作数 operand** | `F64:$lhs` | `toy.add %0, %1` 的 `%0` | **运行时**（前面 op 算出来的） |

**`$名字` 有什么用**？生成 C++ 后会变成访问器方法：

| `.td` 写法 | 生成的 C++ 方法 |
|---|---|
| `F64Attr:$value` | `getValueAttr()` / `getValue()` |
| `F64:$lhs` | `getLhs()` |
| `F64:$rhs` | `getRhs()` |
| `F64:$result` | `getResult()` |

### 4.4 `assemblyFormat` = printf 格式串

```tablegen
let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
```

| 记号 | 含义 |
|---|---|
| `$lhs` | 打印 `lhs` 这个绑定（操作数/属性/结果） |
| `` `,` `` | 原样打印字面量符号（**反引号包住**） |
| `attr-dict` | "剩余属性放这"的占位符（**规范要求每个 op 都得有**） |
| `type($result)` | 打印结果类型 |

**这一个字段就自动生成了 `parse()` 和 `print()` 两个完整函数**——这是 MLIR 帮你省的最大工作量。

### 4.5 `[Pure]` 是什么意思（重点）

`[Pure]` = 给 op 贴**"我是纯函数"的标签**，编译器据此才敢优化。

它等价于声明两件事：

| 标签 | 含义 | C++ 里对应 |
|---|---|---|
| **无副作用** | 不写内存、不打日志、无外部可观察效果 | `MemoryEffectOpInterface`（无 Write） |
| **确定性可折叠** | 输入相同 → 输出一定相同 | `ConditionallySpeculatable` + 恒可投机 |

**标了 Pure，编译器获得的权力**：

| 优化 | 例子 | 不标会怎样 |
|---|---|---|
| 常量折叠（folder） | 编译期算 `toy.add(2.5, 3.5)` → `6.0` | 只能运行时算 |
| 公共子表达式消除（CSE） | 相同计算删一个，共用结果 | 不敢删（万一副作用执行两次呢） |
| 死代码消除（DCE） | 结果没人用就删掉 | 不敢删（万一有副作用呢） |
| 投机执行（speculation） | 提前算、算错就丢弃 | 不能 |

**为什么必须贴标签（和 C 的关键区别）**：

- C 里编译器**能读函数体自己分析**
- MLIR 的 op 是**自定义的、对编译器是黑盒**——它没有函数体可看

所以框架**只能靠你贴的 trait 猜 op 性质**。

> **贴 trait = 给编译器签承诺书**。你说"我是纯的"，它才敢优化；你骗它，程序就会被优化错。

---

## 五、⭐ 亲眼看看 `.td` 变成了什么（本课最震撼的部分）

新手最懵的是"我写的东西到底变成了啥"。下面全部是**真实生成的代码**。

### 5.1 方言：`.td` 20 行 → `.inc` 26 行

**你写的**：
```tablegen
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}
```

**生成的**（`ToyDialect.h.inc` 全文）：
```cpp
namespace mlir {
namespace toy {

class ToyDialect : public ::mlir::Dialect {
  explicit ToyDialect(::mlir::MLIRContext *context);

  void initialize();                          // ← 你要在 .cpp 里实现它
  friend class ::mlir::MLIRContext;
public:
  ~ToyDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("toy");      // ← 来自 let name = "toy"
  }
};
} // namespace toy
} // namespace mlir
```

**对应关系**：
| `.td` | 生成的 C++ |
|---|---|
| `let name = "toy"` | `getDialectNamespace()` 返回 `"toy"` |
| `let cppNamespace = "::mlir::toy"` | `namespace mlir { namespace toy {` |
| `def Toy_Dialect` | `class ToyDialect` |

**`initialize()` 是唯一要你手写实现的**（在 `ToyDialect.cpp` 里注册 op）。

**`ToyDialect.cpp.inc`（25 行，全文节选）**：
```cpp
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::ToyDialect)

ToyDialect::ToyDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ToyDialect>())
{
  initialize();          // ← 构造函数自动调用 initialize()
}

ToyDialect::~ToyDialect() = default;
```

### 5.2 `[Pure]` 变成了什么

**你写的**：
```tablegen
def Toy_ConstantOp : Toy_Op<"constant", [Pure]>
```

**生成的类头**（`ToyOps.h.inc` 真实内容）：
```cpp
class ConstantOp : public ::mlir::Op<ConstantOp,
    ::mlir::OpTrait::ZeroRegions,
    ::mlir::OpTrait::OneResult,
    ::mlir::OpTrait::OneTypedResult<::mlir::FloatType>::Impl,
    ::mlir::OpTrait::ZeroSuccessors,
    ::mlir::OpTrait::ZeroOperands,              // ← 没有操作数（constant）
    ::mlir::OpTrait::OpInvariants,
    ::mlir::BytecodeOpInterface::Trait,         // ← 因为带属性，要序列化
    ::mlir::ConditionallySpeculatable::Trait,   ┐
    ::mlir::OpTrait::AlwaysSpeculatableImplTrait,│ ← 这三个就是 [Pure] 展开的！
    ::mlir::MemoryEffectOpInterface::Trait      ┘
> {
```

**`[Pure]` 一个词，展开成 3 个 C++ trait**：可折叠 + 可投机 + 无内存副作用。

> 这就是"贴标签"的真相——**你写 1 个词，编译器看到 3 个 trait，据此决定怎么优化。**

注意其余 trait 是自动推出来的：`toy.constant` 没有输入（`ZeroOperands`）、有一个结果（`OneResult`）、没有 Region（`ZeroRegions`）——这些**都来自 `.td` 的 arguments/results 声明**，你不用写。

### 5.3 属性变成了什么

**你写的**：
```tablegen
let arguments = (ins F64Attr:$value);
```

**生成的**（`ToyOps.h.inc` 里的 `Properties` 结构，真实内容）：
```cpp
struct Properties {
  using valueTy = ::mlir::FloatAttr;    // ← F64Attr 变成 mlir::FloatAttr
  valueTy value;                        // ← $value 变成成员变量

  auto getValue() {
    auto &propStorage = this->value;
    return ::llvm::cast<::mlir::FloatAttr>(propStorage);
  }
  void setValue(const ::mlir::FloatAttr &propValue) { this->value = propValue; }
  ...
};
```

**访问器是自动生成的**——你写 `$value`，得到 `getValue()`。

### 5.4 ⭐ `assemblyFormat` 变成了什么（含"坑 5"的答案）

**你写的**：
```tablegen
let assemblyFormat = "$value attr-dict";
```

**生成的 `parse`（读 IR 文本）**：
```cpp
ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  ::mlir::FloatAttr valueAttr;

  if (parser.parseCustomAttributeWithFallback(valueAttr, parser.getBuilder().getF64Type())) {
  //     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 用 f64 作为兜底类型！
    return ::mlir::failure();
  }
  if (valueAttr) result.getOrAddProperties<ConstantOp::Properties>().value = valueAttr;
  ...
}
```

**生成的 `print`（写 IR 文本）**：
```cpp
void ConstantOp::print(OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getValueAttr());
  //            ^^^^^^^^^^^^^^^^^^^^^^ 打印时不带类型！
  ...
}
```

**⭐ 这里完美解释了 E 组的"坑 5"**（`toy.constant 2.5` 能过、`2.5 : f64` 反而报错）：

| 方向 | 生成的代码行为 | 结果 |
|---|---|---|
| **读**（parse） | 用 `parseCustomAttributeWithFallback(..., f64类型)` | 写 `2.5` 时自动当 f64 解析；画蛇添足写 `: f64` 反而多余 |
| **写**（print） | `printAttributeWithoutType` | 输出 `2.500000e+00`，**不带** `: f64` |

而 `toy.add` 因为 `assemblyFormat` 里有 `type($result)`，所以**必须带** `: f64`。

> **这就是 `.td` 的威力**：你写一行格式串，工具生成两个函数，连"边界情况"（缺类型怎么办）都替你处理了。

### 5.5 op 列表（`GET_OP_LIST`）

**`ToyOps.cpp.inc` 里的真实内容**：
```cpp
#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::toy::AddOp,
::mlir::toy::ConstantOp,
::mlir::toy::MulOp
#endif  // GET_OP_LIST
```

这个列表**就是给 `addOperations<...>` 用的**（见 6.3）——`tblgen` 遍历 `.td` 里所有 `def Toy_*Op` 自动汇总。

### 5.6 生成命令总表（4 个产物）

| 命令 | 读 | 写 | 行数 |
|---|---|---|---|
| `mlir-tblgen -gen-dialect-decls` | `ToyDialect.td` | `ToyDialect.h.inc` | 26 |
| `mlir-tblgen -gen-dialect-defs` | `ToyDialect.td` | `ToyDialect.cpp.inc` | 25 |
| `mlir-tblgen -gen-op-decls` | `ToyOps.td` | `ToyOps.h.inc` | 522 |
| `mlir-tblgen -gen-op-defs` | `ToyOps.td` | `ToyOps.cpp.inc` | 544 |
| | | **合计** | **1117** |

---

## 六、手写文件（"插头"，为什么那么薄）

现在该明白为什么手写文件只有几行了——**真正的代码都生成了，手写文件只负责"把产物接进编译单元"**。

### 6.1 `ToyDialect.h`（16 行，全文）

```cpp
#ifndef TOY_TOYDIALECT_H
#define TOY_TOYDIALECT_H

#include "mlir/IR/Dialect.h"
#include "Toy/ToyDialect.h.inc"     // ← 把生成的类声明展开进来

#endif
```

**就这么简单。** ⚠️ 注意 `Toy/ToyDialect.h.inc` 这个文件**源码里不存在**——是 CMake 在编译前生成的。

### 6.2 `ToyOps.h`（37 行）——稍复杂，要 include 一堆

```cpp
#include "mlir/Bytecode/BytecodeOpInterface.h"      // 带属性的 op 要序列化（坑 4）
#include "mlir/IR/Builders.h"                       // build() 工厂用 Builder（坑 5）
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"               // parse/print 用 OpAsmParser（坑 5）
#include "mlir/Interfaces/SideEffectInterfaces.h"   // [Pure] 用

#define GET_OP_CLASSES                    // ← 关键宏！
#include "Toy/ToyOps.h.inc"
```

**为什么每行都必须有**：

| include | 缺了会报什么 |
|---|---|
| `BytecodeOpInterface.h` | `'BytecodeOpInterface' is not a member of 'mlir'` |
| `Builders.h` | `invalid use of incomplete type 'class mlir::Builder'` |
| `OpImplementation.h` | `incomplete type 'mlir::OpAsmParser' used in nested name specifier` |
| `SideEffectInterfaces.h` | `[Pure]` trait 展开需要 `MemoryEffectOpInterface` |

**`#define GET_OP_CLASSES` 是什么？**

生成的 `.inc` 文件里同时写着**好几套代码**，用宏开关控制展开哪一套：

```cpp
// ToyOps.h.inc 内部结构（示意）
namespace mlir { namespace toy { class ConstantOp; } }   // ← 永远展开（前置声明）

#ifdef GET_OP_CLASSES
  class ConstantOp : public ::mlir::Op<...> { ... };      // ← 只有定义了宏才展开
#endif
```

**同一个 `.inc` 被 include 两次、干两件事**：
- `ToyDialect.cpp` 里 `#define GET_OP_LIST` → 展开出"op 类型列表"
- `ToyOps.cpp` 里 `#define GET_OP_CLASSES` → 展开出"op 的实现代码"

### 6.3 `ToyDialect.cpp`（24 行）——唯一有"逻辑"的地方

```cpp
#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"

using namespace mlir;
using namespace mlir::toy;

#include "Toy/ToyDialect.cpp.inc"       // 展开生成的构造/析构

void ToyDialect::initialize() {         // ← 实现 .h.inc 里声明的 initialize()
  addOperations<
#define GET_OP_LIST
#include "Toy/ToyOps.cpp.inc"           // ← 展开出 op 类型列表
      >();
}
```

**`initialize()` 干的事**：方言被加载时，把 `toy.add`/`toy.constant`/`toy.mul` **登记进方言**——等价于"告诉 MLIR：toy 方言有这三个 op"。

展开后相当于：
```cpp
addOperations<::mlir::toy::AddOp, ::mlir::toy::ConstantOp, ::mlir::toy::MulOp>();
```

> **宏套 include 的写法**（`addOperations<` 里面塞 `#define` 和 `#include`）看着很怪，这是 LLVM 的老传统。理解成"在这里展开一个列表"就行。

### 6.4 `ToyOps.cpp`（15 行，全文）

```cpp
#include "Toy/ToyOps.h"
#include "Toy/ToyDialect.h"

#define GET_OP_CLASSES
#include "Toy/ToyOps.cpp.inc"     // ← 展开所有 op 的实现
```

**结束。** 因为 parse/print/build/verify 全是生成的，这里没别的事可做。

---

## 七、`edialect-opt.cpp`（入口）

```cpp
#include "Toy/ToyDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"

using namespace mlir;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  DialectRegistry registry;
  registerAllDialects(registry);              // ① 注册所有内置方言
  registry.insert<mlir::toy::ToyDialect>();   // ② 注册我们的 toy 方言 ← 关键一行！

  return asMainReturnCode(MlirOptMain(
      argc, argv, "edialect-opt - a toy dialect demo tool\n", registry));
}
```

**第 ② 行就是"教 MLIR 认识 toy"的开关**——不写这行，遇到 `toy.xxx` 就报 `unregistered dialect`。

**为什么是 `toy::ToyDialect`？** 因为 `.td` 里写了 `cppNamespace = "::mlir::toy"`，且 `def Toy_Dialect` 去掉了 `_Dialect` 后缀。

---

## 八、CMakeLists.txt 讲透

### 8.1 生成规则（`add_custom_command`）

```cmake
set(TBLGEN ${MLIR_BUILD_DIR}/bin/mlir-tblgen)

# 以 op 声明为例（另外 3 个同理）
add_custom_command(
  OUTPUT  ${GEN_DIR}/ToyOps.h.inc                      # 产物
  COMMAND ${TBLGEN} -gen-op-decls ${TD_INCLUDES}       # 命令
          ${CMAKE_CURRENT_SOURCE_DIR}/include/Toy/ToyOps.td
          -o ${GEN_DIR}/ToyOps.h.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/Toy/ToyOps.td)   # 依赖 .td
```

**作用**：告诉 ninja"这个 `.inc` 是这么来的"。改了 `.td`，`.inc` 会自动重新生成（增量构建）。

### 8.2 tblgen 的 include 路径（3 个）

```cmake
set(TD_INCLUDES
  -I ${MLIR_SRC_DIR}/include                      # 找 mlir/IR/OpBase.td
  -I ${MLIR_BUILD_DIR}/tools/mlir/include          # 生成的 MLIR .td
  -I ${CMAKE_CURRENT_SOURCE_DIR}/include)          # 找我们的 Toy/ToyDialect.td
```

### 8.3 ⚠️ 生成路径必须带子目录（坑 2）

```cmake
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen/Toy")          # ← 注意结尾的 /Toy
include_directories(... ${CMAKE_CURRENT_BINARY_DIR}/gen )    # ← 但 include 加父目录
```

**为什么这么绕？** 因为源码里写的是：
```cpp
#include "Toy/ToyOps.h.inc"     // ← 带 Toy/ 前缀
```

所以：
- **生成**要到 `build/gen/Toy/`（保持 `Toy/` 这一层）
- **include 路径**只加 `build/gen`（父目录），这样 `"Toy/ToyOps.h.inc"` 才能解析到

### 8.4 ⚠️ 4 个 `.inc` 都要被依赖（坑 3）

```cmake
add_custom_target(toy-inc DEPENDS
  ${GEN_DIR}/ToyDialect.h.inc
  ${GEN_DIR}/ToyDialect.cpp.inc
  ${GEN_DIR}/ToyOps.h.inc
  ${GEN_DIR}/ToyOps.cpp.inc)

add_dependencies(edialect-opt toy-inc)      # ← 关键：先跑完 tblgen 再编译
```

**为什么不能只写 `add_executable`？**
> 如果只在 `add_executable` 里列 `.h.inc`，ninja 会认为"我只需要这 2 个文件"，**`.cpp.inc` 永远不会被生成**（因为没人依赖它们），编译时报文件不存在。
> 必须建一个 `custom target` 把 4 个全部依赖上。

### 8.5 剩下两个（D 组就踩过）

```cmake
# include 路径（4 个来源，D 组已讲）
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include      # 我们自己的头
  ${LLVM_SRC_DIR}/include                  # LLVM 源码
  ${MLIR_BUILD_DIR}/include                # MLIR 生成
  ${MLIR_BUILD_DIR}/tools/mlir/include
  ${MLIR_SRC_DIR}/include                  # MLIR 源码
  ${CMAKE_CURRENT_BINARY_DIR}/gen)         # 我们的生成目录

# -fno-rtti：MLIR 库关了 RTTI，我们必须一致
set_target_properties(edialect-opt PROPERTIES COMPILE_FLAGS "-fno-rtti")

# 全量链接 + --start-group（静态库循环依赖）
file(GLOB MLIR_LIBS ${MLIR_BUILD_DIR}/lib/libMLIR*.a)
file(GLOB LLVM_LIBS ${MLIR_BUILD_DIR}/lib/libLLVM*.a)
target_link_libraries(edialect-opt
  -Wl,--start-group ${MLIR_LIBS} ${LLVM_LIBS} -Wl,--end-group
  pthread z dl)
```

---

## 九、完整执行流程（编译时发生了什么）

```bash
cd "lesson-23-MLIR入门（未完成）/E-dialect"

# ① 配置（生成 Ninja 构建文件）
cmake -B build -G Ninja .

# ② 编译——注意 ninja 的输出顺序！
ninja -C build
# [1/8] Generating gen/Toy/ToyDialect.h.inc    ← 先跑 mlir-tblgen
# [2/8] Generating gen/Toy/ToyDialect.cpp.inc
# [3/8] Generating gen/Toy/ToyOps.h.inc
# [4/8] Generating gen/Toy/ToyOps.cpp.inc
# [5/8] Building CXX object .../ToyOps.cpp.o   ← 再编译手写文件
# [6/8] Building CXX object .../ToyDialect.cpp.o
# [7/8] Building CXX object .../edialect-opt.cpp.o
# [8/8] Linking CXX executable edialect-opt

# ③ 跑！用 toy 方言写的程序
./build/edialect-opt test.mlir
```

**输出**：
```mlir
module {
  func.func @main() -> f64 {
    %0 = toy.constant 2.500000e+00     // 2.5 被格式化成科学计数法
    %1 = toy.constant 3.500000e+00
    %2 = toy.add %0, %1 : f64          // 2.5+3.5 = 6.0
    %3 = toy.mul %0, %2 : f64          // 2.5*6.0 = 15.0
    return %3 : f64
  }
}
```

---

## 十、常用命令速查

### 编译

```bash
cd "lesson-23-MLIR入门（未完成）/E-dialect"

# 首次（或目录改名后）
rm -rf build && cmake -B build -G Ninja . && ninja -C build

# 改了 .td 或 .cpp 后
ninja -C build          # tblgen + 编译自动增量
```

### 4 个验证实验

```bash
# ① 解析 + 原样打印
./build/edialect-opt test.mlir

# ② 看底层真实结构（generic 格式）
./build/edialect-opt test.mlir --mlir-print-op-generic
#   %0 = "toy.constant"() <{value = 2.500000e+00 : f64}> : () -> f64
#   %2 = "toy.add"(%0, %1) : (f64, f64) -> f64

# ③ verify 抓错（故意写错类型）
printf 'module {\n func.func @main() -> f64 {\n  %%0 = toy.constant 2.5\n  %%1 = toy.add %%0, %%0 : i32\n  return %%1 : f64\n }\n}\n' > /tmp/bad.mlir
./build/edialect-opt /tmp/bad.mlir
# error: custom op 'toy.add' invalid kind of type specified

# ④ 未注册方言拒绝（对照实验）
printf 'module {\n func.func @main() {\n  %%0 = "unk.weird"() : () -> ()\n  return\n }\n}\n' > /tmp/unk.mlir
./build/edialect-opt /tmp/unk.mlir
# error: operation being parsed with an unregistered dialect.

# ⑤ 看帮助（MlirOptMain 白送的一堆选项）
./build/edialect-opt --help
```

### 单独跑 mlir-tblgen（不通过 ninja，观察生成过程）

```bash
TBLGEN=../../mlir-src/build/bin/mlir-tblgen
INC="-I ../../mlir-src/mlir/include -I ../../mlir-src/build/tools/mlir/include -I include"

$TBLGEN -gen-op-decls $INC include/Toy/ToyOps.td -o /tmp/ToyOps.h.inc
head -40 /tmp/ToyOps.h.inc      # 亲眼看生成结果
```

---

## 十一、疑问沉淀（学习过程问过的问题）

### 11.1 MLIR 用尖括号 `< >` 传参啊？

**不是 MLIR，是 TableGen。**

- `.td` 用的是 TableGen 语法（LLVM 自研的小语言，有自己的解析器）
- MLIR 只是**用** TableGen 来声明方言/op
- 尖括号传参是 TableGen 的方式，**类型和值混着传**
- 写法模仿 C++ 模板是**故意的**（让你眼熟），但它不是 C++

### 11.2 `.td` 语法完全没见过，好难

**别把它当编程语言学**。记住：
1. 全部语法只有 4 条（include / class / def / let）
2. 其余符号（`Op`/`Dialect`/`F64Attr`/`Pure`）是**素材库**，当标准库用
3. **90% 的 op 定义都是同一个骨架**——会抄、会改名字就够用

### 11.3 `[Pure]` 是什么意思

见 §4.5。一句话：**给 op 贴"我是纯函数"的标签，编译器据此才敢做折叠/CSE/DCE**。

### 11.4 为什么手写的 C++ 那么薄

因为**真正干活的代码都生成了**（1117 行 `.inc`）。手写文件只是"插头"——`#include` 产物 + 实现唯一一个手写函数（`initialize()`）。

### 11.5 生成的 `.inc` 为什么能 include 两次

因为它用**宏开关**分块：`GET_OP_LIST` 展开出 op 列表，`GET_OP_CLASSES` 展开出 op 实现。不定义宏时，只展开前置声明。

---

## 十二、踩坑记录（E 组 5 个坑，全验证过）

### 坑 1：`[Pure]` 报 `Variable not defined`

```
error: Variable not defined: 'Pure'
def Toy_ConstantOp : Toy_Op<"constant", [Pure]> {
                                         ^
```
**原因**：`Pure` 定义在 `mlir/Interfaces/SideEffectInterfaces.td`，`OpBase.td` 的 include 链**不带它**。
**解法**：
```tablegen
include "mlir/Interfaces/SideEffectInterfaces.td"
```

### 坑 2：`.inc` 找不到

```
fatal error: Toy/ToyOps.h.inc: No such file or directory
```
**原因**：生成到了 `build/gen/`，但源码 include `"Toy/ToyOps.h.inc"`。
**解法**：`GEN_DIR` 改成 `build/gen/Toy`，`include_directories` 加 `build/gen`（父目录）。

### 坑 3：`.cpp.inc` 从不生成

```
fatal error: Toy/ToyOps.cpp.inc: No such file or directory
```
**原因**：只把 `.h.inc` 列进 `add_executable`，ninja 认为其他的不需要。
**解法**：`add_custom_target(toy-inc DEPENDS 全部4个)` + `add_dependencies(edialect-opt toy-inc)`。

### 坑 4：`BytecodeOpInterface` 找不到

```
error: 'BytecodeOpInterface' is not a member of 'mlir'
```
**原因**：带 attribute 的 op 会生成 Properties 序列化代码（MLIR 19 的 properties 机制），需要该接口定义。
**解法**：`ToyOps.h` include `mlir/Bytecode/BytecodeOpInterface.h`。

### 坑 5：`toy.constant 2.5 : f64` 报错

```
error: expected operation name in quotes
    %0 = toy.constant 2.5 : f64
```
**原因**：`F64Attr` 的 parse 用 `parseCustomAttributeWithFallback(..., f64)` 兜底，print 用 `printAttributeWithoutType`——**不带类型**。
**解法**：`toy.constant 2.5`（不带 `: f64`）；而 `toy.add`/`toy.mul` 因为 `assemblyFormat` 有 `type($result)`，**必须带** `: f64`。

### 坑 6（环境类）：目录改名后构建失败

```
ninja: error: rebuilding 'build.ninja': subcommand failed
/usr/local/bin/cmake --regenerate-during-build -S .../lesson-23-MLIR入门/E-dialect ...
```
**原因**：`lesson-23-MLIR入门` 改名为 `lesson-23-MLIR入门（未完成）` 后，`build/` 里 CMake cache 存的还是旧路径。
**解法**：`rm -rf build && cmake -B build -G Ninja . && ninja -C build`。

---

## 十三、零件速查表

### 文件分工

| 文件 | 谁写 | 作用 |
|---|---|---|
| `ToyDialect.td` | 你 | 方言声明（名字/命名空间） |
| `ToyOps.td` | 你 | op 声明（输入/输出/格式/特性） |
| `ToyDialect.h` | 你 | include `ToyDialect.h.inc` |
| `ToyOps.h` | 你 | `GET_OP_CLASSES` + include `ToyOps.h.inc` |
| `ToyDialect.cpp` | 你 | `initialize()` 注册 op |
| `ToyOps.cpp` | 你 | `GET_OP_CLASSES` + include `ToyOps.cpp.inc` |
| `edialect-opt.cpp` | 你 | `registry.insert<ToyDialect>()` |
| `CMakeLists.txt` | 你 | 4 条生成规则 + 依赖 |
| `*.inc`（4个） | 工具 | 真正的 C++ 代码 |

### `.td` 语法

| 语法 | 用途 |
|---|---|
| `include "x.td"` | 引入定义（**显式，无传递性**） |
| `class X<参数> : 父类;` | 可复用模板 |
| `def X : 模板<参数> { ... }` | 具体记录 |
| `let 字段 = 值;` | 赋值 |

### op 定义字段

| 字段 | 含义 |
|---|---|
| `arguments (ins ...)` | 输入（属性 / 操作数） |
| `results (outs ...)` | 输出（类型） |
| `assemblyFormat` | IR 文本格式（自动生成 parse/print） |
| `summary` / `description` | 文档（可选） |
| `traits`（模板参数） | 特性标签（`Pure` 等） |

### `assemblyFormat` 记号

| 记号 | 含义 |
|---|---|
| `$name` | 打印绑定 |
| `` `字面量` `` | 原样打印符号 |
| `attr-dict` | 剩余属性占位（**必须有**） |
| `type($x)` | 打印类型 |
| `(` `)` | 分组，括号内成对出现 |

### mlir-tblgen 命令

| 命令 | 产物 |
|---|---|
| `-gen-dialect-decls` | `ToyDialect.h.inc` |
| `-gen-dialect-defs` | `ToyDialect.cpp.inc` |
| `-gen-op-decls` | `ToyOps.h.inc` |
| `-gen-op-defs` | `ToyOps.cpp.inc` |

---

## 十四、和已有知识的打通

| 你会的 | E 组对应 | 说明 |
|---|---|---|
| MNN 算子注册 | `addOperations<>` + `registry.insert<>` | 都是"登记新算子/方言" |
| 此前 NPU 宏/脚本 → asm | `.td` → mlir-tblgen → `.inc` | 描述驱动 + 代码生成，**同一个模式** |
| TVM 的 Relay Op 注册 | ODS/TableGen | TVM 用手写 C++，MLIR 用声明式 |
| ONNX 算子定义（schema） | `.td` 里的 `arguments`/`results` | 都是"算子签名声明" |
| C++ 模板 | TableGen `class` + `<>` | 语法像，但不是一回事 |
| C 的 `#include` | TableGen `include` | 区别：**无语义传递性** |

---

## 十五、套路总结——以后加新 op 怎么做

假设要加 `toy.sub`（减法），**只改 1 处 + 重编**：

**① `ToyOps.td` 末尾加**：
```tablegen
def Toy_SubOp : Toy_Op<"sub", [Pure]> {
  let summary = "Subtract two f64 values";
  let arguments = (ins F64:$lhs, F64:$rhs);
  let results   = (outs F64:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

**② 重新编译**：
```bash
ninja -C build     # tblgen 自动重新生成 4 个 .inc
```

**③ 测试**：
```mlir
%4 = toy.sub %0, %1 : f64
```

**不需要改**：`ToyDialect.cpp`（`GET_OP_LIST` 自动包含新 op）、任何 `.h`、`CMakeLists.txt`。

> **这就是"描述驱动"的威力**——加 op 只写声明，其余全自动。

---

## 十六、小结

### 三句话记住 E 组

1. **你写说明书（`.td`），`mlir-tblgen` 写代码（`.inc`），你再用插头（`.h`/`.cpp`）接进工具。**
2. **TableGen 只有 4 条语法，op 定义都是同一个骨架，会抄就行。**
3. **`[Pure]` 之类的 trait 是"给编译器签承诺书"——它才敢优化你的 op。**

### 全流程复盘（一张图）

```
你写 ToyOps.td ──┐
                 │  ninja 调用 mlir-tblgen
                 ▼
        build/gen/Toy/ToyOps.h.inc   （522 行 C++ 类声明）
        build/gen/Toy/ToyOps.cpp.inc （544 行 parse/print/build）
                 │
                 │  被 #include 展开
                 ▼
   ToyOps.h ──► ToyOps.cpp ──► 编译成 .o
                                   │
   ToyDialect.td ──► 同理 ─────────┤
                                   ▼
                          edialect-opt 可执行文件
                                   │
                                   ▼
                        能解析 toy.constant / toy.add / toy.mul
```

### 下一步可以学什么

E 组现在只做到"**能解析自己的方言**"。自然的下一步：

| 方向 | 内容 | 对应经验 |
|---|---|---|
| **F 组** | 给 `toy.add` 写 **folder**（常量折叠）+ lower 到 `arith` | 把此前 NPU 的"宏→asm"链条在 MLIR 里重建 |
| **linalg** | 张量计算高层抽象（tiling/pack/fusion） | TVM schedule / 手工切 MAC 阵列块 |
| 更远 | bufferization（tensor→memref）、量化、IREE、NPU 后端 | 片上特征/权重缓冲规划自动化 |
