# Lesson 24：推理工程（Inference Engineering）

> 本课讲的是**云侧推理服务工程**——怎么把生成式模型在生产环境跑得又快又便宜又稳。
> 不是芯片工具链，也不是边缘部署，而是**数据中心 GPU 上的服务侧工程**。
>
> 这一课在你的技能版图里补的是**云侧视角**：edge 部署（此前 NPU / 某 DSP/NPU / MNN）你已经很熟，但"云上怎么服务"是面试和宽带晋升里的另外半边。

---

## 一、这一课从哪来

| 项目 | 说明 |
|---|---|
| 原书 | *Inference Engineering*，Philip Kiely 著，Baseten Books 出版，2026 年 1 月定稿 |
| 规模 | 8 章 + 2 附录，259 页，ISBN `979-8-9943597-2-3` |
| 免费正式版 | <https://www.baseten.co/inference-engineering/digital-download/>（填表邮件送达 PDF/EPUB/有声书） |
| 交互伴读 | <https://learn-inference.com/>（本地离线副本见仓库根 `learn-inference/`） |
| 本地全文 | 用 `pdftotext` 抽出的纯文本在 `tmp/ie_full.txt`（`tmp/` 已被 gitignore，**不要提交**） |

**本课是什么**：按主题重组的学习笔记 + 讲解 + 可跑脚本，是**我读完之后的重新组织**，不是原书内容的搬运。
**本课不是什么**：原书的替代品。原书细节远多于本课，重要章节建议回原文读一遍。

### ⚠️ 版权

原书版权归 Baseten Labs（"All rights reserved"），作者免费公开电子版但要遵守站点声明的 `ai-train=no`。

- 本课**只包含我自己的总结、讲解、表格和代码**，不复述原文段落
- 所有引用原文数字（如 H100 的 989 TFLOPS）都是**事实性数据**，可自由使用
- 不要在仓库里放原书 PDF / EPUB / 大段原文摘录，也不要把本课当成原书的再分发
- 学有余力请去支持作者，原书写得很扎实

---

## 二、为什么值得单独开一课

对照你现有的技术版图，这一课补三块：

| 你的短板（职业规划里列的） | 本课对应章节 | 补到什么程度 |
|---|---|---|
| **1. 量化理论** | [06](06_量化_动态范围与粒度.md) | 云侧术语体系（dynamic range / granularity / 敏感度排序），和 Lesson 21/22 的公式侧互补 |
| **2. 编译器源码级理解** | 补不了 | 这课是应用层，靠 MLIR（Lesson 23）|
| **3. 性能建模** | [03](03_模型机制与瓶颈计算.md) | roofline 完整推导 + 可跑脚本，把你从"经验驱动"推到"模型驱动" |

另外还有一层价值：**面试时"edge + cloud 两头都懂"是加分项**。
你有 此前 NPU 的 MAC 利用率/bank 冲突/DMA 双缓冲，有 某 DSP/NPU 的 RTF 0.044，这些是**硬件侧直觉**；
但被问到"你怎么决定该优化搬运还是优化 kernel""量化到 FP8 还是 FP4""什么时候 batching 没用"这类**建模问题**时，需要的是这一课的定量语言。

---

## 三、怎么用这一课

### 阅读路线（按你的优先级排序，不是按书的顺序）

| 优先级 | 章节 | 理由 |
|---|---|---|
| 🔴 **精读** | [03 模型与瓶颈计算](03_模型机制与瓶颈计算.md) | 补性能建模短板，全课的定量核心 |
| 🔴 **精读** | [03a Prefill 与 Decode 详解](03a_Prefill与Decode详解.md) | ⭐ **03 章的慢放版**：卡在 prefill/decode 上就看这份（由浅入深 5 层） |
| 🔴 **精读** | [06 量化](06_量化_动态范围与粒度.md) | 补量化理论短板，和 Lesson 21/22 打通 |
| 🟡 值得读 | [05 软件栈](05_软件栈_CUDA到推理引擎.md) | kernel 融合/kernel 选择 = 你在 此前 NPU 干的手工活 |
| 🟡 值得读 | [04 硬件](04_硬件_GPU与本地推理.md) | 3.5 节点名 Qualcomm Hexagon，正是你的领域 |
| 🟡 值得读 | [08 多模态](08_多模态_语音与生成.md) | ASR/TTS 的流式与分块，和 KWS/流式降噪 可比 |
| 🟢 泛读 | [01 全景](01_全景_三层与六技术.md) [02 前置决策](02_前置决策_度量与模型选择.md) | 建立全局框架，快速过 |
| 🟢 泛读 | [07 技术（投机/缓存/并行/拆解）](07_投机_缓存_并行_拆解.md) | 云侧高频词，知道是什么即可 |
| ⚪ **可跳过** | [09 生产部署](09_生产部署.md) | 云上运维/采购，与你的方向关系不大 |

### 配合脚本动手

```bash
cd lesson-24-推理工程（Inference Engineering）

python3 prefill_vs_decode.py            # ⭐ 先看这个：复用率视角，讲透为什么 prefill/decode 瓶颈不同
python3 prefill_vs_decode.py --trace    # 一次请求的完整流程：prefill 一次 + decode N 次
python3 measure_ridge.py                # ⭐ 实测【你这台机器】的 ridge —— 上面几个都是纸面复算，这个是真尺子
python3 roofline.py                     # 各代 GPU 的 ridge + 你的 NPU 对比
python3 roofline.py --gpu H100          # 单卡详细：未融合 vs 融合 attention 的算术强度
python3 quant_demo.py                   # 动态范围 vs 粒度：误差为什么这么分布
```

> 💡 **`measure_ridge.py` 和其他几个的区别**：其他脚本算的是**公式的输出**，它测的是**硬件的表现**。
> 本机（i9-14900K）实测：带宽 50.8 GB/s、FP32 算力 1546 GFLOPS、**ridge 30.4 ops/byte**。
> 想测你的板子，改 `--custom` 那类参数即可。

---

## 四、章节索引

| # | 文件 | 对应原书 | 一句话 |
|---|---|---|---|
| 00 | [README.md](README.md) | — | 课程总览（本文件） |
| 01 | [全景：三层与六技术](01_全景_三层与六技术.md) | Ch0 | 全书骨架：runtime/infrastructure/tooling + 六个技术 |
| 02 | [前置决策：度量与模型选择](02_前置决策_度量与模型选择.md) | Ch1 | 动手之前先想清楚：TTFT/TPS、百分位、共享 vs 专用 |
| 03 | [模型机制与瓶颈计算](03_模型机制与瓶颈计算.md) | Ch2 | ⭐ roofline 推导：prefill 算力受限、decode 带宽受限 |
| 03a | [Prefill 与 Decode 详解](03a_Prefill与Decode详解.md) | Ch2+Ch5.5 | ⭐ **专题慢放版**：五层递进，从"为什么要分两段"推到"芯片为什么做两套"；含本机 ridge 实测 |
| 04 | [硬件：GPU 与本地推理](04_硬件_GPU与本地推理.md) | Ch3 | 读懂 spec sheet；3.5 节 = 你的主场 |
| 05 | [软件栈：CUDA 到推理引擎](05_软件栈_CUDA到推理引擎.md) | Ch4 | kernel 融合、safetensors vs ONNX、三大引擎 |
| 06 | [量化：动态范围与粒度](06_量化_动态范围与粒度.md) | Ch5.1 | ⭐ 浮点为何胜整数；量化顺序与质量验证 |
| 07 | [投机、缓存、并行、拆解](07_投机_缓存_并行_拆解.md) | Ch5.2-5.5 | 四个提性能手段及各自代价 |
| 08 | [多模态：语音与生成](08_多模态_语音与生成.md) | Ch6 | VLM/embedding/ASR/TTS/图像/视频 |
| 09 | [生产部署](09_生产部署.md) | Ch7 | 容器、扩缩、多云、可观测（可跳过） |
| 10 | [术语速查与延伸阅读](10_术语速查与延伸阅读.md) | App A/B | 中英对照术语 + 论文清单 |

---

## 五、与既有课程的呼应

| 本课内容 | 呼应 | 打通点 |
|---|---|---|
| [03](03_模型机制与瓶颈计算.md) roofline | **技术名词表 11.3 / 11.6** | 你的公式卡里已经有 $I$ 和 $I_{ridge}$，本课补上"怎么用"和可跑代码 |
| [03](03_模型机制与瓶颈计算.md) prefill/decode | **技术名词表 11.2** | TTFT ↔ prefill ↔ 算力受限；TPOT ↔ decode ↔ 带宽受限 |
| [03](03_模型机制与瓶颈计算.md) kernel 融合 | **Lesson 20**（MNN 手写 NEON） | 融合 = 减少中间结果往返内存，和你手写 NEON 是同一件事 |
| [05](05_软件栈_CUDA到推理引擎.md) kernel 选择 | **此前 NPU 手工调度** | 你在 此前 NPU 干的就是"为特定硬件选/写 kernel" |
| [05](05_软件栈_CUDA到推理引擎.md) ONNX/TensorRT | **Lesson 01-12**（ORT 自定义算子） | 本课给出行业判断：中间表示路线正在被"引擎直连"分流 |
| [06](06_量化_动态范围与粒度.md) 量化 | **Lesson 21 / 22** | Lesson 21 是公式侧（scale/zero_point），本课是格式侧（FP8/MXFP8/NVFP4） |
| [06](06_量化_动态范围与粒度.md) 量化 | **Lesson 22 实验C** | MNN 的 `alpha` per-channel scale ↔ 本课的 granularity |
| [08](08_多模态_语音与生成.md) ASR | **KWS / 流式降噪（此前的工作语音线）** | 200ms 往返目标、VAD 分块、RTF 度量 |
| [04](04_硬件_GPU与本地推理.md) 移动推理 | **移动 DSP/NPU + QNN** | 原书点名 Qualcomm Hexagon 作为移动加速器 |
| [04](04_硬件_GPU与本地推理.md) 其他加速器 | **此前 NPU** | 原书列的"专用 ASIC 下注方向"可以对照 此前 NPU 定位 |

---

## 六、一页速览（给未来的自己）

**全书的两个框架**

```
三层：runtime（榨干单卡） → infrastructure（跨集群扩容） → tooling（抽象层次）

六技术（Ch5 目录）：batching · caching · quantization · speculation · parallelism · disaggregation
```

**全书的定量核心（一句话）**

$$\text{算 } I = \frac{\text{FLOPs}}{\text{Bytes}} \quad\Longrightarrow\quad \text{比 } I_{ridge} = \frac{\text{Peak FLOPs/s}}{\text{Bandwidth}} \quad\Longrightarrow\quad \text{低于平衡点就闭眼优化搬运}$$

**三条必背结论**

| 场景 | 瓶颈 | 该做什么 |
|---|---|---|
| LLM **prefill** | compute bound | 降精度、更好的 kernel |
| LLM **decode** | memory bound | **batching**、量化、投机、KV cache 复用 |
| **图像/视频生成** | compute bound | 降精度、更好的 kernel、更少步数（**batching 几乎没用**） |

**两个最容易误答的点**

1. **batching 不会让单个用户变快** —— 它让你已经在付的带宽做更多活（code 侧：提升 throughput，不降 latency）
2. **decode 的 $I$ 远小于 ridge 是设计事实不是实现缺陷** —— 70B FP16 每 token 要搬 140GB，H100 上光读取就 42ms，所以 batching 是唯一出路

---

*上一课：[Lesson 23 MLIR 入门](../lesson-23-MLIR入门（未完成）/) ｜ 相关：[技术名词表 十一节](../技术名词表.md)*
