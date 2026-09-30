# Inference Engineering Notes

一份**自下而上贯通推理栈**的学习笔记：从推理运行时（ONNX Runtime）→ AI 编译器（TVM / MNN / MLIR）→ 部署与量化 → 性能建模。

每个 lesson 都是**可亲手跑通的最小实验**：带 `Makefile` 或单文件脚本，代码、模型生成脚本、说明文档齐备，尽量做到「一节课 = 一个可复现的结论」。

> 📌 本仓库为个人学习记录。涉及工作经历的部分已做**通用化处理**（不包含具体公司、产品型号与项目名称）。

---

## 为什么有这么一份笔记

推理工程师常见的困境是「工具会用，但说不清底层在发生什么」：

- 写自定义算子时，不清楚计算图怎么落到设备上；
- 调性能时，凭经验试参数，而不是先判断瓶颈在算力还是带宽；
- 遇到量化，只知道调 `scale`，不清楚误差从哪来、怎么传播。

这份笔记的目标是把每一层的**因果关系**串起来 —— 运行时怎么调度、编译器怎么 lowering、硬件在等什么、误差如何累积。

---

## 结构

### 一、推理运行时：ONNX Runtime（lesson 01–12）

| 课 | 主题 |
|---|---|
| 01 | 最小自定义算子（算子注册、kernel 实现） |
| 02 | 打包成动态库给 Python 加载 |
| 03 | 带属性的算子 + 形状推断（shape inference） |
| 04 | 多种数据类型（int32 / int64 / float / double） |
| 05 | 多输入多输出 |
| 06 | 带属性的 ReduceMean 与形状推理 |
| 07 | SessionOptions 与图优化控制 |
| 08 | C API 推理（脱离 C++ 封装） |
| 09 | Session 内省（读图结构、算子分布） |
| 10 | Session 保存与加载（优化后模型的序列化） |
| 11 | 性能分析与调优（profiling、算子耗时归因） |
| 12 | I/O Binding 零拷贝（避免输入输出搬运） |

### 二、AI 编译器：TVM（lesson 13–18）

| 课 | 主题 |
|---|---|
| 13 | AI 编译器入门：IR / 调度 / codegen 全景 |
| 14 | AutoTVM 自动调优（搜索空间 + 测量） |
| 15 | 部署：Relay → 计算图执行器 |
| 16 | 部署 C API |
| 17 | TensorIR 与 TVMScript（显式写 tile 与布局） |
| 18 | 交叉编译到 ARM（需开发板） |

### 三、端侧推理引擎：MNN（lesson 19–22）

| 课 | 主题 |
|---|---|
| 19 | MNN 初识与构建（需开发板） |
| 20 | 手写 NEON vs TVM 生成代码对比（需开发板） |
| 21 | 量化原理与误差分析（对称/非对称、定标法、误差累积） |
| 22 | MNN 算子实现与量化部署 |

### 四、编译器框架与推理工程

| 目录 | 主题 |
|---|---|
| `lesson-23-MLIR入门` | MLIR：dialect、pass、ODS/TableGen、自定义方言与 pass 工程（进行中） |
| `lesson-24-推理工程` | 推理工程全景：两阶段（prefill/decode）、三层结构、roofline、量化粒度、多模态、生产部署 |
| `lesson-25-浮点格式与私有数据格式` | 浮点格式（fp16/bf16/fp8/私有格式）与数值表示 |
| `lookup-table-approx` | 查表近似专题：分段线性 LUT、幂分解、激活函数适用性 |
| `技术名词表.md` | 个人术语整理：把工程口语翻译成工业界通用术语 |

---

## 快速开始

各课独立，一般流程：

```bash
cd lesson-XX-xxx
cat SUMMARY.md      # 或 STEPS.md / README.md —— 先看这一课要解决什么问题
make                # 或 python3 <script>.py
```

### 环境

- **ONNX Runtime 相关**：`ort-bin/` 内含预编译的头文件与 pkgconfig（`.so` 不入库，请自备或使用环境变量 `ORT_DIR`）
- **TVM / MNN / MLIR 相关**：源码树**不随仓库分发**（体积过大），请自行克隆：
  ```bash
  git clone --recursive https://github.com/apache/tvm tvm-src
  git clone https://github.com/alibaba/MNN mnn-src
  git clone https://github.com/llvm/llvm-project mlir-src
  ```
- **交叉编译**：需要 Android NDK，通过环境变量指定根目录：
  ```bash
  export NDK=$HOME/toolchain/android-ndk-rXX
  ```
- **Python**：见各课 `requirements.txt`

### 路径约定

脚本中的路径用环境变量而非硬编码，便于在任意目录克隆后直接运行：

| 变量 | 含义 |
|---|---|
| `$REPO` | 本仓库根目录 |
| `$HOME/toolchain/` | 本地工具链（NDK 等） |
| `NDK` | Android NDK 根目录 |
| `ORT_DIR` | ONNX Runtime 根目录 |

---

## 关于 `learn-inference/`

`learn-inference/` 是 [learn-inference.com](https://learn-inference.com/) 的**离线阅读副本**（Philip Kiely《Inference Engineering》，Baseten）。该站正文免费公开，声明 `ai-train=no`。

- 保留副本的用途：**离线阅读与 grep**，交互模拟器需本地起服务（`./learn-inference/serve.sh`）。
- 版权归原作者。若需完整版本，官方提供[免费 PDF/EPUB/有声书申请](https://www.baseten.co/inference-engineering/digital-download/)。
- 已包含官方 PDF 与站点镜像，便于离线通读；如需从仓库移除，取消 `.gitignore` 中 `# learn-inference/` 的注释即可。

---

## 约定

- 每个结论尽量给出**可复现的证据**：能跑的实验、能算的公式、能引的数据手册。
- 不确定的内容明确标注为「推测 / 待验证」，不写成事实（`技术名词表.md` 中有专门章节记录待核项）。
- 涉及具体产品的性能数字，只引**公开规格**并注明来源。

---

## 许可

笔记与代码仅供学习交流。第三方资料（ONNX Runtime 头文件、书籍副本等）版权归各自所有者。
