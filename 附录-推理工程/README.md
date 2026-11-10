# Learn Inference 离线资料

[learn-inference.com](https://learn-inference.com/) 的本地离线副本 —— Philip Kiely《Inference Engineering》(Baseten) 的**交互式伴读**，主题是**云侧 GPU 上服务生成式模型**，不是芯片工具链。

> 抓取时间：2026-09-16
> 站点 robots.txt：`Allow: /`（`Content-Signal: search=yes, ai-input=yes, ai-train=no`）

## ⚠️ 版权与使用范围

- 这是**版权内容**，作者把正文免费公开在 learn-inference.com，但要遵守站点声明的 `ai-train=no`。
- 本目录**仅供本地个人学习**，**不要提交到 git、不要分发**。
- 已在仓库根 `.gitignore` 里加了 `learn-inference/`，防止误传到 `<owner>/<repo>` 公开仓库。
- 如果你想支持作者，官方还提供免费 PDF / EPUB / 有声书，**填表即邮件送达**：
  - 申请表：<https://www.baseten.co/inference-engineering/digital-download/>
  - 在线读原书（免费，无需表单）：<https://www.baseten.co/inference-engineering/book/>
  - 纸质版（付费）：<https://books.baseten.com/products/inference-engineering>

### 申请 PDF / EPUB 的具体步骤

表单只有 4 个字段，`*` 为必填：

| 字段 | 必填 | 说明 |
|---|---|---|
| First name | ✅ | 英文名或拼音均可 |
| Last name | ✅ | 同上 |
| Company | ❌ | 可以留空 |
| Email | ✅ | **PDF + EPUB + 有声书的下载链接发到这里** |

1. 打开 <https://www.baseten.co/inference-engineering/digital-download/>
2. 填好上面 4 项，点 **Get the book**
3. 提交后跳转到 `.../digital-download/thank-you`，页面提示 *"You will receive a PDF and EPUB copy of the book by email shortly."*
4. 去邮箱收信（**记得翻垃圾邮件**），邮件里是 PDF / EPUB / 有声书链接

注意事项：

- 这是 **HubSpot 表单**（portal `22114337`），本质是线索收集 → 大概率会把你加进 Baseten 的营销邮件列表。建议用一个你不在意收营销邮件的邮箱，或收到后用邮件底部的退订。
- 表单由 HubSpot 托管，海外服务，国内网络可能加载慢或需要代理；表单页打不开时可以用原书的免费在线阅读版。
- 我没有代填这个表单：它需要你的真实姓名和邮箱，而且提交是真实副作用（会订阅邮件）。请自己填更稳妥。

## 目录结构

```
learn-inference/
├── md/                     ← 推荐日常阅读用这个
│   ├── llms-full.md        ← 全书合并版单文件（181KB），可直接 grep
│   ├── llms-index.md       ← 官方索引，含每章一句话摘要
│   └── site/               ← 56 个页面的官方 Markdown，按章节分目录
├── site/                   ← 完整站点镜像（含 JS/CSS/字体，交互图表可跑）
│   └── learn-inference.com/
├── tools/
│   └── relativize_assets.py  ← 把绝对资源路径改相对（已执行过）
├── serve.sh                ← 起本地服务读镜像
└── README.md
```

**两种读法：**

| 目的 | 用什么 |
|---|---|
| 快速读、grep、抄术语 | `md/` 下的 Markdown |
| 玩交互模拟器（拖滑块看延迟/吞吐曲线） | `./serve.sh` 然后开 <http://127.0.0.1:8123/learn-inference.com/index.html> |

> Markdown 里图表是**占位符 + 链接**（官方就这么设计：图教的是"输入改变后的响应"，写下来就丢信息）。要体验模拟器就回镜像站点。

已验证：镜像经本地 http 服务打开时 CSS/JS 正常加载、零控制台报错、GPU 档位切换等交互组件可点可算。

## 内容规模

| 项目 | 数量 |
|---|---|
| Markdown 页面 | 56 |
| 站点资源 | 16 JS / 2 CSS / 20 字体，约 3.2MB |
| 总体积 | 8.4MB |

章节：0 Inference / 1 Prerequisites / 2 Models / 3 Hardware / 4 Software / 5 Techniques / 6 Modalities / 7 Production + Glossary + Further reading。

## 学习索引（按"云侧视角补充"的定位）

本资料在整体学习路线里是**云侧视角补充**，不是主线（主线是 TVM → MNN → MLIR）。

**值得精读：**

- `md/site/index.md`、`chapters/inference/two-phases.md`、`chapters/inference/three-layers.md`
  → 第 0 章全景：TTFT/TPOT 两阶段、runtime/infrastructure/tooling 三层
- `chapters/models/bottlenecks.md` → **2.4 计算推理瓶颈**，roofline / 算术强度，补"性能建模"短板（可交互算，推荐在镜像站点里拖着玩）
- `chapters/techniques/quantization.md` → **5.1 量化**，重点是 *dynamic range vs granularity* 这组术语（per-tensor / per-channel / per-block 32 / NVFP4 16），补"量化粒度"这块的云侧表述
- `chapters/software/benchmarking.md` → 4.5 benchmark 与压测
- `chapters/techniques/speculative-decoding.md`、`chapters/techniques/caching.md`
  → 投机解码、KV cache / prefix caching，LLM 岗高频词
- `chapters/glossary.md` → 术语表，和仓库根的 `技术名词表.md` 对照着收词

**可跳过：** 第 7 章云上运维（容器化 / 自动扩缩 / 多云采购）、3.3 实例选型、4.4 NVIDIA Dynamo、3.2 GPU 代际史。

**面试价值：** edge 部署（你的 MNN/TVM/某 DSP/NPU 背景）+ cloud serving 两头都懂是加分项；TTFT/TPOT、continuous batching、prefix caching 这些词要能张口就来。

## 重新抓取 / 更新

站点每个页面在 URL 后加 `.md` 就是 Markdown，全文合并在 `llms-full.txt`：

```bash
cd learn-inference
curl -s https://learn-inference.com/llms-full.txt -o md/llms-full.md
curl -s https://learn-inference.com/llms.txt      -o md/llms-index.md
curl -s https://learn-inference.com/sitemap.xml | grep -o '<loc>[^<]*</loc>' | sed 's/<[^>]*>//g'   # 页面清单
```

整站镜像（约 160 个文件）：

```bash
cd learn-inference/site
wget -e robots=on -r -l inf -k -p -E -np --wait=0.3 --random-wait https://learn-inference.com/
cd .. && python3 tools/relativize_assets.py
```
