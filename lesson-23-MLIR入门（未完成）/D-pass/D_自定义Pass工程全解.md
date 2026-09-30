# Lesson 23-D：用 C++ 写自定义 Pass（工程全解）

## 〇、本课定位（先读）

C 组你用现成的 pass（canonicalize/cse）观察优化。D 组**亲手写一个 pass**——用 C++ 实现"常量折叠"（C 组 canonicalize 里折叠功能的迷你版），编译成自己的工具 `dpass-opt`。

**这是从"用 MLIR"到"写编译器"的分水岭**。C 组是"用别人的工具"，D 组是"造自己的工具"——这正是工具链工程师的日常。

```
输入 test.mlir：                       输出：
%c = arith.addi %a, %b   (2+3)   →    %c = arith.constant 5   （addi 没了）
```

**本课为什么突然变复杂**：
| 之前（A/B/C） | 现在（D） |
|---|---|
| 只写 `.mlir` 文本 | 要写 **C++ 代码** |
| 用现成 `mlir-opt` | 要**自己编译**一个工具 |
| 不用管链接 | 要**链接 MLIR/LLVM 库**（坑最多的地方） |

**心理预期**：D 组是 12 课规划里最难的一课，踩坑多、概念新，但跨过去你就真正"写过编译器代码"了。

---

## 一、工程结构

```
lesson-23-MLIR入门/D-pass/
├── CMakeLists.txt    # 构建配置（4 个 include + -fno-rtti + 全量链接）
├── Passes.h          # pass 声明（给别的文件看的"接口"）
├── Passes.cpp        # pass 实现（核心：折叠逻辑 + 工厂 + 注册）
├── dpass-opt.cpp     # 入口 main（注册方言 + 注册 pass + 启动主循环）
├── test.mlir         # 测试输入
└── build/            # 构建产物（dpass-opt 在这）
```

**各文件职责**：
```
Passes.cpp  ← 你的优化逻辑（核心）＝加工机器
Passes.h    ← 声明"有 createFoldAddiPass 和 registerDPassPasses 两个函数"＝机器接口
dpass-opt.cpp ← 程序入口，把 pass 注册进一个"迷你 mlir-opt"＝工厂调度员
CMakeLists.txt ← 告诉编译器"怎么把上面这些编成可执行文件"＝施工图纸
test.mlir   ← 喂给工具跑的输入＝原材料
```

**数据流**：`test.mlir → dpass-opt 读入 → Passes.cpp 的规则处理 → 输出优化后 IR`

---

## 二、核心概念：pass 的三个组成

一个自定义 pass = **类定义 + 工厂函数 + 注册函数** 三件套。

### 2.1 类定义（Passes.cpp 的 struct）

```cpp
struct FoldAddiPass
    : public PassWrapper<FoldAddiPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const final { return "dpass-fold-addi"; }
  void runOnOperation() override { ... }   // 核心逻辑
};
```

**三个关键点**：
1. **继承 PassWrapper** = "FoldAddiPass 是一种 Pass"，自动获得 pass 的一切能力
   （类比 `Dog : Animal` = "Dog 是一种 Animal"）
2. **模板参数**：
   - `FoldAddiPass`（自己）→ CRTP，让基类知道子类是谁
   - `OperationPass<func::FuncOp>` → 这个 pass 处理"函数"
3. **`runOnOperation()`** → pass 的入口，MLIR 对每个函数调用一次

**为什么继承这么写**：MLIR 的 pass 框架用模板实现多态。你只要"填"它要求的函数（runOnOperation），框架就替你管好"什么时候调用、怎么进管线"。

### 2.2 工厂函数

```cpp
std::unique_ptr<mlir::Pass> createFoldAddiPass() {
  return std::make_unique<FoldAddiPass>();
}
```

**为什么需要**：注册 pass 时要给它一个"能创建实例的函数"（工厂模式，像"会做这道菜的师傅"）。`unique_ptr` 是智能指针，自动管理内存（用完自动释放，不用手动 delete）。

### 2.3 注册函数

```cpp
void registerDPassPasses() {
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createFoldAddiPass(); });
}
```

**为什么需要**：`dpass-opt` 启动时调用它，把 pass"写进系统菜单"，让 `--pass-pipeline="...dpass-fold-addi..."` 能用。没有注册，pass 只是个类，没人知道它存在。

---

## 三、核心逻辑逐行讲（runOnOperation 内部）

整体结构（一层套一层）：
```
namespace dpass {              ← 命名空间（防名字冲突）
  struct FoldAddiPass {        ← pass 类定义
    runOnOperation() {         ← 入口（MLIR 对每个函数调一次）
      func.walk(lambda) {      ← 遍历（每遇到 addi 回调一次）
        ...折叠逻辑...         ← 真正干活
      }
    }
  };
}
```

### 3.1 遍历：func.walk

```cpp
func::FuncOp func = getOperation();  // 拿到当前函数
func.walk([&](arith::AddIOp add) { ... });  // 遍历所有 addi
```

- **walk = 深度优先遍历函数里的所有操作**，每遇到一个 `arith.addi` 调用一次 lambda（回调函数）
- **类比**：C 里 `for (节点 : 图) { 处理(节点); }`
- **lambda = 匿名函数**：`[&]` 按引用捕获外部变量（能用外面的变量）；`(arith::AddIOp add)` 是参数——walk 把当前遇到的 addi 传进来
- **lambda 和普通 static 函数的核心区别**：lambda 能捕获外部变量

### 3.2 匹配：getDefiningOp

```cpp
auto lhs = add.getLhs().getDefiningOp<arith::ConstantOp>();
auto rhs = add.getRhs().getDefiningOp<arith::ConstantOp>();
if (!lhs || !rhs) return;   // 有操作数不是常量，跳过
```

| 代码 | 含义 | 类比 |
|---|---|---|
| `add.getLhs()` | addi 的左操作数 | `%c = addi %a, %b` 里的 `%a` |
| `.getDefiningOp<ConstantOp>()` | 问"定义 %a 的操作是 constant 吗" | `isConstant(%a)` |
| `if (!lhs \|\| !rhs) return` | 有操作数不是常量就跳过 | 只有 `2+3` 能折叠，`x+3` 不能 |

### 3.3 取值：dyn_cast + getInt

```cpp
auto lval = lhs.getValue().dyn_cast<IntegerAttr>().getInt();
auto rval = rhs.getValue().dyn_cast<IntegerAttr>().getInt();
auto result = lval + rval;   // 编译期就算出来！
```

| 代码 | 含义 |
|---|---|
| `lhs.getValue()` | 常量的属性值（`arith.constant 5` 里的 5） |
| `.dyn_cast<IntegerAttr>()` | 转成整数属性（安全转换，非整数返回 null） |
| `.getInt()` | 取出整数（IntegerAttr → 5） |

**这就是"折叠"**——编译期把 `2+3` 算成 `5`，运行时不用算了。

### 3.4 替换：OpBuilder + replaceAllUsesWith + erase

```cpp
OpBuilder builder(add);
auto newConst = builder.create<arith::ConstantOp>(
    add.getLoc(), add.getType(), builder.getIntegerAttr(add.getType(), result));
add.replaceAllUsesWith(newConst.getResult());
add.erase();
```

**三步**（顺序很重要！）：
1. **create**：在 addi 位置造新常量 `constant(5)`（OpBuilder = 能造新操作的"建造器"）
2. **replaceAllUsesWith**：所有用 addi 结果的地方，改用新常量（通用"改接线"工具，靠 MLIR 的 use-def 链）
3. **erase**：删掉原 addi

**为什么顺序不能乱**：必须先替换所有使用点，才能安全删除 addi。如果先删，用 addi 结果的地方就悬空了（引用不存在的值）。

**类比**：
```
原来：%c = addi(2,3) → return %c
① create：%new = constant(5)
② replace：return 改用 %new（return %new）
③ erase：删掉 %c = addi(2,3)
结果：%new = constant(5) → return %new
```

**注意**："create 新的 → replace 使用 → erase 旧的"是替换**任何** op 的通用四步，不限于折叠。

---

## 四、CMakeLists.txt 讲透（坑最多的地方）

### 4.1 include 路径（4 个，缺一不可）

```cmake
include_directories(
  ${LLVM_SRC_DIR}/include              # LLVM 源码头文件（llvm/Support/Casting.h 等）
  ${MLIR_BUILD_DIR}/include            # LLVM 生成头文件
  ${MLIR_BUILD_DIR}/tools/mlir/include # MLIR 生成头文件（TableGen 产物，如 ArithOps.h.inc）
  ${MLIR_SRC_DIR}/include              # MLIR 源码头文件
)
```

**为什么 4 个**：MLIR 的头文件分散在源码和构建产物两处，缺一个就报 `not found`。

### 4.2 -fno-rtti（必须加！）

```cmake
COMPILE_FLAGS "-fno-rtti"
```

**RTTI = 运行时类型信息**（C++ 的 `dynamic_cast`/`typeid` 依赖它）。LLVM/MLIR 库编译时用了 `-fno-rtti`（为了性能）。

**为什么必须关**：**链接某个库，编译选项必须和那个库一致**。我们开着 RTTI，MLIR 关了，链接报 `undefined reference to typeinfo for mlir::Pass`。

### 4.3 全量链接 + --start-group

```cmake
file(GLOB MLIR_LIBS ${MLIR_BUILD_DIR}/lib/libMLIR*.a)
file(GLOB LLVM_LIBS ${MLIR_BUILD_DIR}/lib/libLLVM*.a)
target_link_libraries(dpass-opt
  -Wl,--start-group ${MLIR_LIBS} ${LLVM_LIBS} -Wl,--end-group
  pthread z dl)
```

| 部分 | 作用 |
|---|---|
| `file(GLOB ...)` | 收集所有 `.a` 静态库（上百个） |
| `--start-group ... --end-group` | 让链接器**反复扫描**这组库，解决循环依赖 |
| `pthread z dl` | 系统库：线程 / 压缩 / 动态加载 |

**为什么需要 --start-group**：MLIR/LLVM 上百个静态库互相依赖（循环），普通顺序链接会报 `undefined reference`。`--start-group` 让链接器像转圈一样来回找符号。

---

## 五、dpass-opt.cpp（入口）

```cpp
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  DialectRegistry registry;
  registerAllDialects(registry);       // ① 注册所有内置方言（都能读）
  dpass::registerDPassPasses();        // ② 注册我们的 pass（唯一私货）
  return asMainReturnCode(MlirOptMain(  // ③ 现成主循环接管
      argc, argv, "dpass-opt - a toy mlir-opt with custom pass\n", registry));
}
```

**本质**：**借 MLIR 的壳，加我们的芯**——复用官方 mlir-opt 的主循环（`MlirOptMain`），只多注册了我们的 pass。

**类比**：官方 mlir-opt 是"大超市"，dpass-opt 是"小超市"，商品（方言）一样，多一个自产商品（dpass-fold-addi）。以后给自家 NPU 写工具，套路一样：换掉 ② 里的 pass 即可。

---

## 六、完整执行流程

```
你运行：dpass-opt --pass-pipeline="...dpass-fold-addi..." test.mlir

1. dpass-opt.cpp 启动 → registerDPassPasses() → 注册表有 "dpass-fold-addi"
2. MLIR 读入 test.mlir → 内存里生成 IR 图
3. pass manager 跑管线 → 查到 dpass-fold-addi → 调创建函数现造实例
4. 对每个函数调 runOnOperation()：
   a. func.walk 遍历所有 addi
   b. 第一个 addi：2+3，都是常量 → 折叠 → 打印 "folded 2 + 3 = 5"
   c. 第二个 addi：5+5，都是常量 → 折叠 → 打印 "folded 5 + 5 = 10"
5. 输出优化后的 IR（只剩 constant 10）
6. 跑完 → unique_ptr 自动销毁 pass 实例
```

**你的 pass 在第 4 步被调用**——`runOnOperation` 里的逻辑就是编译器在这个函数上做的优化。

---

## 七、常用命令速查

### run.sh 一键脚本

每行是一个命令 + 注释，想看哪个效果就**取消哪行的注释**再 `./run.sh`。

### 手动命令

```bash
# 编译（改了 C++ 代码后）
cmake -B build -G Ninja .    # 首次
ninja -C build               # 之后增量编译

# 1. 标准折叠（推荐）：指定 pass 管线 + 输入文件
./build/dpass-opt --pass-pipeline="builtin.module(func.func(dpass-fold-addi))" test.mlir

# 2. 只看帮助
./build/dpass-opt --help

# 3. 只解析不优化（不指定 pipeline，原样打印 IR）
./build/dpass-opt test.mlir

# 4. 不带文件参数（从键盘 stdin 读，敲完 Ctrl-D 结束）
./build/dpass-opt
```

### 运行输出预期

```bash
# [dpass] folded 2 + 3 = 5
# [dpass] folded 5 + 5 = 10
# module {
#   func.func @main() -> i32 {
#     %c10_i32 = arith.constant 10 : i32
#     return %c10_i32 : i32
#   }
# }
```

### 传参规则

```
./build/dpass-opt --pass-pipeline="..." test.mlir
│               │                         │
argv[0] 程序名   argv[1] 第一个参数         argv[2] 第二个参数
```
- 命令按**空格**切段 → 每段一个参数；引号 `"..."` 内**不切分**（是一个整体）
- `argv[0]` 永远是程序名；`argc` = 参数个数（含 argv[0]）
- **main 只是收快递**，真正解析参数的是 `MlirOptMain`（main 把 argc/argv 原样传给它）

---

## 八、疑问沉淀（学习过程问过的问题）

### 8.1 插件 vs 静态编译（两种注册方式）

| | .so 插件（ORT 自定义算子 / MLIR 插件） | 我们 D 组（可执行文件） |
|---|---|---|
| 形态 | 动态库，无 main | 可执行文件，有 main |
| 谁调用它 | 框架 dlopen 后按**固定符号**找 | **main 里手动调用** |
| 固定入口 | `RegisterCustomOps` / `mlirGetPassPluginInfo` | 无（自己调自己） |
| 能改名吗 | 不能 | 能（同步改 main 即可） |

**记 ORT 时"提前规定好名字"是 .so 插件世界**；D 组是静态编译，`registerDPassPasses` 是 main 手动调的普通函数。

### 8.2 重载 vs 重写 vs 普通函数

| | 重载 overload | 重写 override | 工厂/注册函数 |
|---|---|---|---|
| 函数名 | 相同 | 相同 | 不同 |
| 参数 | 不同 | 相同 | 无 |
| 关系 | 平行函数 | 子类重写基类 virtual | 无继承 |

**真正的"重写"在 `runOnOperation() override`**（重写 Pass 基类虚函数，框架按它自动调，**不能改名**）。`createFoldAddiPass`/`registerDPassPasses` 只是普通函数 + 调用关系，**随便改名**。

### 8.3 手动常量折叠 vs canonicalize 自动折叠

| | C 组 canonicalize | D 组手写 pass |
|---|---|---|
| 谁找 addi | 框架 | 自己 walk |
| 谁判断/计算/替换 | op 自带 fold 方法 | 自己 getDefiningOp + 算 + create/replace/erase |
| 本质 | "用别人的折叠" | "自己造折叠" |

### 8.4 walk 的 add 参数是什么

```cpp
func.walk([&](arith::AddIOp add) { ... });
```
`add` 是 **walk 传给回调的参数**（当前遍历到的 addi）——像 C 的 `for(op:图){处理(op);}` 里传给处理函数的 op。类型 `arith::AddIOp` 告诉 walk"只对 addi 感兴趣"，其他操作自动跳过。

### 8.5 -fno-rtti 和 --start-group 是什么

- `-fno-rtti`：关闭运行时类型信息（dynamic_cast/typeid 不能用了），**为了和 MLIR 库编译选项一致**，否则链接报 typeinfo 错误
- `--start-group`：让链接器反复扫描一组库，**解决静态库循环依赖**（MLIR/LLVM 上百个库互相依赖）

---

## 九、踩坑记录（D 组 8 个坑，全验证过）

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | `libMLIRPass.a missing` | CMake 路径 `../../../mlir-src` 多了 1 级 | 改成 `../../mlir-src` |
| 2 | `llvm/Support/Casting.h not found` | 缺 LLVM include 路径 | 加 `${LLVM_SRC_DIR}/include` |
| 3 | `registerDPassPasses is not a member` | Passes.h 没声明 | 补声明 |
| 4 | `PassRegistration<FoldAddiPass>()` 用法错 | 新版 MLIR 改 `registerPass` | 用 `mlir::registerPass` |
| 5 | `unique_ptr` 无法转换 | FoldAddiPass 在匿名 namespace + 缺 FuncOps.h | 移出匿名 namespace + include FuncOps.h |
| 6 | 链接缺 `inferShrS`/`itaniumDemangle` | MLIR/LLVM 库有循环依赖 | `--start-group` 全量链接 |
| 7 | 插件 dlopen 失败 | 插件方案符号解析脆弱 | 改用**自编译完整 dpass-opt**（最可靠） |
| 8 | `typeinfo for mlir::Pass` 未定义 | MLIR 库是 `-fno-rtti` 编译的 | 编译加 `-fno-rtti` |

**核心教训**（90% 的人卡在这）：
1. **include 4 个路径缺一不可**
2. **`-fno-rtti` 必须加**（和 MLIR 库一致）
3. **全量链接 + `--start-group`** 处理循环依赖
4. **插件方案（--load-pass-plugin）脆弱**，自编译完整工具最可靠

---

## 十、零件速查表

| 代码 | 是什么 | 类比 |
|---|---|---|
| `struct FoldAddiPass : public PassWrapper<...>` | 定义 pass 类（继承基类） | "我是一个 Pass" |
| `getArgument()` | pass 的名字 | 工牌上的名字 |
| `runOnOperation()` | pass 的入口 | 主函数 |
| `getOperation()` | 当前处理的对象（函数） | 拿到函数 |
| `func.walk(lambda)` | 遍历所有操作 | for 循环遍历图 |
| `add.getLhs().getDefiningOp<ConstantOp>()` | 左操作数是常量吗 | isConstant(左) |
| `getValue().dyn_cast<IntegerAttr>().getInt()` | 取常量的整数值 | 取出数值 |
| `OpBuilder + create` | 造新常量 | 生成新指令 |
| `replaceAllUsesWith` | 使用点改用新值 | 替换引用 |
| `erase()` | 删掉旧操作 | 删除指令 |
| `createFoldAddiPass()` | 工厂 | 造 pass 的流水线 |
| `registerDPassPasses()` | 注册 | 告诉 MLIR"有这号 pass" |
| `-fno-rtti` | 关运行时类型信息 | 和 MLIR 库一致 |
| `--start-group` | 解决循环依赖 | 反复扫描库 |

---

## 十一、和已有知识的打通

| 你会的 | 对应 D 组 |
|---|---|
| C 组 canonicalize 的折叠 | 你的 pass 实现了它的迷你版 |
| 此前 NPU 手写算子（遍历/匹配/生成） | `walk` + `getDefiningOp` + `create` 就是自动化版 |
| TVM 写 pass | 同一个概念，只是 API 不同 |
| ORT 自定义算子注册 | .so 插件注册 vs 本课静态注册的对比（见 8.1） |

---

## 十二、小结

**这一课你做了什么**：
1. ✅ 用 C++ 写了一个 pass 类（继承 PassWrapper + runOnOperation）
2. ✅ 实现了"常量折叠"逻辑（walk → 匹配 → 替换 → 删除）
3. ✅ 建了 CMake 工程，链接 MLIR/LLVM 库（踩了 8 个坑）
4. ✅ 编译出自己的 dpass-opt 工具，跑通折叠（2+3→5, 5+5→10）

**你跨过了什么**：从"用 MLIR 命令行"到"写编译器代码"。工具链岗位的日常工作就是写 pass——你现在有这个能力了。
