# Lesson 11 附：ORT Profile JSON 解读指南

> 以 `profile.json_2026-08-03_13-46-44_786.json`（`./main profile 1` 生成）为示例
> 环境：ORT 1.27.0，3 层 MLP 模型，4 线程，跑 1 次推理（+1 预热）

---

## 1. 这是什么格式

**Chrome Trace 格式**（`chrome://tracing` / [Perfetto](https://ui.perfetto.dev) 的输入格式）。
顶层是一个**事件数组**（ORT 1.27+ 直接是 `[...]`，旧版是 `{"traceEvents": [...]}`）。

每个事件的关键字段：

| 字段 | 含义 |
|------|------|
| `cat` | 类别：`Node`=算子执行 / `Session`=会话阶段 |
| `name` | 事件名：`xxx_kernel_time`（带节点名前缀） |
| `dur` | 持续时间（**微秒 us**，不是毫秒！） |
| `ts` | 开始时间戳（us，相对 profiling 开始） |
| `args.op_name` | 算子类型（分析热点**必须用它**，不是 name） |
| `args.node_index` | 优化后图中的节点序号 |
| `args.thread_scheduling_stats` | 线程调度详情（多线程是否生效） |

---

## 2. 示例文件的行号索引（只跑 1 次 = 14 条事件）

```
行  9   model_loading_uri       dur=1746  ← 读模型文件 (一次性)
行 19   session_initialization  dur=2365  ← 建图+图优化 (一次性)

──── 第 1 次推理 = 预热 (冷启动) ────
行 29   fused Gemm1  dur=55     ← 冷启动, 慢 5 倍 (op_name=FusedGemm, 行56)
行 96   fused Gemm2  dur=21
行 163  Gemm3       dur=3
行 228  Softmax1    dur=9
行 288  SequentialExecutor  dur=98   ← 整次执行区间
行 298  model_run   dur=107          ← 整次推理总耗时

──── 第 2 次推理 = 正式 (热了) ────
行 308  fused Gemm1  dur=10     ← 55 → 10, 预热效果明显
行 375  fused Gemm2  dur=13
行 442  Gemm3       dur=2
行 507  Softmax1    dur=2
行 567  SequentialExecutor  dur=34
行 577  model_run   dur=38     ← 107 → 38, 这才是真实性能
```

**最核心的 6 行：**

| 行号 | 是什么 | 值 | 结论 |
|------|--------|-----|------|
| 行 29 | 第 1 次 Gemm1 | 55 us | 冷启动慢 ~5 倍 |
| 行 308 | 第 2 次 Gemm1 | 10 us | 预热后才是真实耗时 |
| 行 298 | 第 1 次 model_run | 107 us | 含冷启动，不可信 |
| 行 577 | 第 2 次 model_run | **38 us** | **真实单次推理耗时** |
| 行 56 | op_name | FusedGemm | Gemm1/2 已被算子融合 |
| 行 288→567 | SequentialExecutor | 98→34 | 每次推理的执行区间 |

---

## 3. 怎么读：三步法

### 3.1 看整体时间线（`ts` + `dur`）

```
ts=1     dur=1746  model_loading_uri
ts=1750  dur=2365  session_initialization   ← 这两条是一次性的, 不计入推理耗时
ts=4129  dur=55    fused Gemm1 (预热, 慢)
ts=4235  dur=10    fused Gemm1 (正式, 快)   ← 之后稳定
```

> **前几次 `dur` 明显偏大 = 冷启动**。session 创建、内存池、kernel 首次编译都是一次性开销。

### 3.2 找热点（`args.op_name` 聚合）

| op_name | 第 2 次推理 dur | 说明 |
|---------|----------------|------|
| FusedGemm (Gemm1+Gemm2) | 10 + 13 = 23 us | **热点，~61%** |
| Gemm | 2 us | 未融合（输出接 Softmax） |
| Softmax | 2 us | 几乎可忽略 |

> 注意：**必须用 `args.op_name` 聚合**，不能用 `name`——`name` 带节点前缀（`fused Gemm1_kernel_time`），每个都不重复，聚合无意义。

### 3.3 看线程是否生效（`thread_scheduling_stats`）

```json
"main_thread": { "core": 14, "Run": 6, "Wait": 0 },
"sub_threads": {
    "...120": { "num_run": 3, "core": 6 },
    "...416": { "num_run": 3, "core": 4 },
    "...712": { "num_run": 3, "core": 25 }
}
```

**1 主 + 3 子 = 4 线程配置确实生效**，Gemm 被切块并行。`num_run` 逐次递增 = 同一节点执行多次。

---

## 4. 性能账本（示例数据的结论）

```
真实单次推理 (行 577):  38 us
  ├─ 算子计算:  10+13+2+2 = 27 us   (71%)
  └─ 搬运开销:  38 - 27   = 11 us   (29%)  ← IoBinding 要优化的部分
```

优化方向：

- **算子耗时 (71%)** → int8 量化 / 图优化 / 更大模型收益更明显
- **搬运开销 (29%)** → **IoBinding 预分配 buffer、复用内存**（Lesson 12）

---

## 5. 分析方法（命令行）

```bash
# ① 用脚本聚合热点
python3 analyze_profile.py profile.json_*.json
# 输出: total events / Top 10 热点算子 / total kernel time

# ② 看时间线瀑布图 (推荐)
# Chrome: 打开 chrome://tracing → Load → 选 json
# 或在线: https://ui.perfetto.dev → Open trace file

# ③ 手动快速查某行
grep -n '"name"\|"dur"\|"op_name"' profile.json_*.json
```

---

## 6. 常见误区

| 误区 | 正解 |
|------|------|
| "这么多事件 = 这么多节点？" | 事件 = **节点 × 执行次数**。跑 N 次推理，每个节点出现 N 条 |
| "用 `name` 聚合热点" | 用 `args.op_name`，`name` 带节点前缀无法合并 |
| "`dur` 单位是毫秒" | 是**微秒**（us），1000 us = 1 ms |
| "第一次推理的耗时可信" | 不可信，冷启动。看预热后的稳定值 |
| "顶层是 dict" | 1.27+ 是**数组**，脚本要兼容两种 |

---

## 7. 实操建议

- **profile 次数**：`./main profile 1` 就够看结构；`profile 20` 用于统计稳定热点
- **跑法对比**：先 `profile 1` 理解结构，再 `profile 20` 出统计结论
- **和 bench 配合**：profile 看"哪里慢"，bench 看"改完快多少"

```json
[
    {
        "cat": "Session",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 1746,
        "ts": 1,
        "ph": "X",
        "name": "model_loading_uri",
        "args": {}
    },
    {
        "cat": "Session",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 2365,
        "ts": 1750,
        "ph": "X",
        "name": "session_initialization",
        "args": {}
    },
    {
        "cat": "Node",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 55,
        "ts": 4129,
        "ph": "X",
        "name": "fused Gemm1_kernel_time",
        "args": {
            "node_index": "6",
            "provider": "CPUExecutionProvider",
            "activation_size": "1024",
            "input_type_shape": [
                {
                    "float": [
                        1,
                        256
                    ]
                },
                {
                    "float": [
                        1024
                    ]
                }
            ],
            "output_type_shape": [
                {
                    "float": [
                        1,
                        1024
                    ]
                }
            ],
            "output_size": "4096",
            "op_name": "FusedGemm",
            "thread_scheduling_stats": {
                "main_thread": {
                    "thread_pool_name": "session-1-intra-op",
                    "thread_id": "140424732347264",
                    "block_size": [
                        1
                    ],
                    "core": 14,
                    "Distribution": 1,
                    "DistributionEnqueue": 0,
                    "Run": 14,
                    "Wait": 6,
                    "WaitRevoke": 0
                },
                "sub_threads": {
                    "140424700229120": {
                        "num_run": 1,
                        "core": 6
                    },
                    "140424691836416": {
                        "num_run": 1,
                        "core": 4
                    },
                    "140424683443712": {
                        "num_run": 1,
                        "core": 25
                    }
                }
            },
            "parameter_size": "4096"
        }
    },
    {
        "cat": "Node",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 21,
        "ts": 4186,
        "ph": "X",
        "name": "fused Gemm2_kernel_time",
        "args": {
            "op_name": "FusedGemm",
            "thread_scheduling_stats": {
                "main_thread": {
                    "thread_pool_name": "session-1-intra-op",
                    "thread_id": "140424732347264",
                    "block_size": [
                        1
                    ],
                    "core": 14,
                    "Distribution": 0,
                    "DistributionEnqueue": 0,
                    "Run": 6,
                    "Wait": 10,
                    "WaitRevoke": 0
                },
                "sub_threads": {
                    "140424700229120": {
                        "num_run": 2,
                        "core": 6
                    },
                    "140424691836416": {
                        "num_run": 2,
                        "core": 4
                    },
                    "140424683443712": {
                        "num_run": 2,
                        "core": 25
                    }
                }
            },
            "parameter_size": "1024",
            "activation_size": "4096",
            "node_index": "7",
            "provider": "CPUExecutionProvider",
            "output_size": "1024",
            "input_type_shape": [
                {
                    "float": [
                        1,
                        1024
                    ]
                },
                {
                    "float": [
                        256
                    ]
                }
            ],
            "output_type_shape": [
                {
                    "float": [
                        1,
                        256
                    ]
                }
            ]
        }
    },
    {
        "cat": "Node",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 3,
        "ts": 4210,
        "ph": "X",
        "name": "Gemm3_kernel_time",
        "args": {
            "op_name": "Gemm",
            "activation_size": "1024",
            "node_index": "4",
            "input_type_shape": [
                {
                    "float": [
                        1,
                        256
                    ]
                },
                {
                    "float": [
                        10
                    ]
                }
            ],
            "output_type_shape": [
                {
                    "float": [
                        1,
                        10
                    ]
                }
            ],
            "provider": "CPUExecutionProvider",
            "output_size": "40",
            "thread_scheduling_stats": {
                "main_thread": {
                    "thread_pool_name": "session-1-intra-op",
                    "thread_id": "140424732347264",
                    "block_size": [],
                    "core": 14,
                    "Distribution": 0,
                    "DistributionEnqueue": 0,
                    "Run": 0,
                    "Wait": 0,
                    "WaitRevoke": 0
                },
                "sub_threads": {
                    "140424700229120": {
                        "num_run": 2,
                        "core": 6
                    },
                    "140424691836416": {
                        "num_run": 2,
                        "core": 4
                    },
                    "140424683443712": {
                        "num_run": 2,
                        "core": 25
                    }
                }
            },
            "parameter_size": "40"
        }
    },
    {
        "cat": "Node",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 9,
        "ts": 4215,
        "ph": "X",
        "name": "Softmax1_kernel_time",
        "args": {
            "op_name": "Softmax",
            "node_index": "5",
            "activation_size": "40",
            "output_type_shape": [
                {
                    "float": [
                        1,
                        10
                    ]
                }
            ],
            "parameter_size": "0",
            "thread_scheduling_stats": {
                "main_thread": {
                    "thread_pool_name": "session-1-intra-op",
                    "thread_id": "140424732347264",
                    "block_size": [],
                    "core": 14,
                    "Distribution": 0,
                    "DistributionEnqueue": 0,
                    "Run": 0,
                    "Wait": 0,
                    "WaitRevoke": 0
                },
                "sub_threads": {
                    "140424700229120": {
                        "num_run": 2,
                        "core": 6
                    },
                    "140424691836416": {
                        "num_run": 2,
                        "core": 4
                    },
                    "140424683443712": {
                        "num_run": 2,
                        "core": 25
                    }
                }
            },
            "input_type_shape": [
                {
                    "float": [
                        1,
                        10
                    ]
                }
            ],
            "provider": "CPUExecutionProvider",
            "output_size": "40"
        }
    },
    {
        "cat": "Session",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 98,
        "ts": 4128,
        "ph": "X",
        "name": "SequentialExecutor::Execute",
        "args": {}
    },
    {
        "cat": "Session",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 107,
        "ts": 4122,
        "ph": "X",
        "name": "model_run",
        "args": {}
    },
    {
        "cat": "Node",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 10,
        "ts": 4235,
        "ph": "X",
        "name": "fused Gemm1_kernel_time",
        "args": {
            "activation_size": "1024",
            "node_index": "6",
            "op_name": "FusedGemm",
            "output_size": "4096",
            "parameter_size": "4096",
            "input_type_shape": [
                {
                    "float": [
                        1,
                        256
                    ]
                },
                {
                    "float": [
                        1024
                    ]
                }
            ],
            "thread_scheduling_stats": {
                "main_thread": {
                    "thread_pool_name": "session-1-intra-op",
                    "thread_id": "140424732347264",
                    "block_size": [
                        1
                    ],
                    "core": 14,
                    "Distribution": 0,
                    "DistributionEnqueue": 0,
                    "Run": 6,
                    "Wait": 0,
                    "WaitRevoke": 0
                },
                "sub_threads": {
                    "140424700229120": {
                        "num_run": 3,
                        "core": 6
                    },
                    "140424691836416": {
                        "num_run": 3,
                        "core": 4
                    },
                    "140424683443712": {
                        "num_run": 3,
                        "core": 25
                    }
                }
            },
            "output_type_shape": [
                {
                    "float": [
                        1,
                        1024
                    ]
                }
            ],
            "provider": "CPUExecutionProvider"
        }
    },
    {
        "cat": "Node",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 13,
        "ts": 4247,
        "ph": "X",
        "name": "fused Gemm2_kernel_time",
        "args": {
            "output_size": "1024",
            "op_name": "FusedGemm",
            "thread_scheduling_stats": {
                "main_thread": {
                    "thread_pool_name": "session-1-intra-op",
                    "thread_id": "140424732347264",
                    "block_size": [
                        1
                    ],
                    "core": 14,
                    "Distribution": 0,
                    "DistributionEnqueue": 0,
                    "Run": 3,
                    "Wait": 5,
                    "WaitRevoke": 0
                },
                "sub_threads": {
                    "140424700229120": {
                        "num_run": 4,
                        "core": 6
                    },
                    "140424691836416": {
                        "num_run": 4,
                        "core": 4
                    },
                    "140424683443712": {
                        "num_run": 4,
                        "core": 25
                    }
                }
            },
            "parameter_size": "1024",
            "activation_size": "4096",
            "input_type_shape": [
                {
                    "float": [
                        1,
                        1024
                    ]
                },
                {
                    "float": [
                        256
                    ]
                }
            ],
            "node_index": "7",
            "output_type_shape": [
                {
                    "float": [
                        1,
                        256
                    ]
                }
            ],
            "provider": "CPUExecutionProvider"
        }
    },
    {
        "cat": "Node",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 2,
        "ts": 4262,
        "ph": "X",
        "name": "Gemm3_kernel_time",
        "args": {
            "parameter_size": "40",
            "provider": "CPUExecutionProvider",
            "node_index": "4",
            "activation_size": "1024",
            "thread_scheduling_stats": {
                "main_thread": {
                    "thread_pool_name": "session-1-intra-op",
                    "thread_id": "140424732347264",
                    "block_size": [],
                    "core": 14,
                    "Distribution": 0,
                    "DistributionEnqueue": 0,
                    "Run": 0,
                    "Wait": 0,
                    "WaitRevoke": 0
                },
                "sub_threads": {
                    "140424700229120": {
                        "num_run": 4,
                        "core": 6
                    },
                    "140424691836416": {
                        "num_run": 4,
                        "core": 4
                    },
                    "140424683443712": {
                        "num_run": 4,
                        "core": 25
                    }
                }
            },
            "input_type_shape": [
                {
                    "float": [
                        1,
                        256
                    ]
                },
                {
                    "float": [
                        10
                    ]
                }
            ],
            "output_size": "40",
            "output_type_shape": [
                {
                    "float": [
                        1,
                        10
                    ]
                }
            ],
            "op_name": "Gemm"
        }
    },
    {
        "cat": "Node",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 2,
        "ts": 4266,
        "ph": "X",
        "name": "Softmax1_kernel_time",
        "args": {
            "node_index": "5",
            "input_type_shape": [
                {
                    "float": [
                        1,
                        10
                    ]
                }
            ],
            "output_size": "40",
            "activation_size": "40",
            "op_name": "Softmax",
            "output_type_shape": [
                {
                    "float": [
                        1,
                        10
                    ]
                }
            ],
            "thread_scheduling_stats": {
                "main_thread": {
                    "thread_pool_name": "session-1-intra-op",
                    "thread_id": "140424732347264",
                    "block_size": [],
                    "core": 14,
                    "Distribution": 0,
                    "DistributionEnqueue": 0,
                    "Run": 0,
                    "Wait": 0,
                    "WaitRevoke": 0
                },
                "sub_threads": {
                    "140424700229120": {
                        "num_run": 4,
                        "core": 6
                    },
                    "140424691836416": {
                        "num_run": 4,
                        "core": 4
                    },
                    "140424683443712": {
                        "num_run": 4,
                        "core": 25
                    }
                }
            },
            "provider": "CPUExecutionProvider",
            "parameter_size": "0"
        }
    },
    {
        "cat": "Session",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 34,
        "ts": 4235,
        "ph": "X",
        "name": "SequentialExecutor::Execute",
        "args": {}
    },
    {
        "cat": "Session",
        "pid": 1967460,
        "tid": 1967460,
        "dur": 38,
        "ts": 4231,
        "ph": "X",
        "name": "model_run",
        "args": {}
    }
]
```