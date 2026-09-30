# session_options_demo.py
# Lesson 07：SessionOptions 与图优化控制
#
# 学完这课你应该会：
#   1. 看懂并配置 SessionOptions 各项参数
#   2. 控制图优化级别并观察对速度的影响
#   3. 查看当前环境的 Providers
#   4. 开关内存池、控制线程数

import time
import numpy as np
import onnxruntime as ort

# ─── 1. 基础：什么都不配，直接用 ───
print("=" * 60)
print("1. 默认 Session（什么 Options 都不配）")
print("=" * 60)

sess = ort.InferenceSession("my_test_model.onnx")
x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
y = sess.run(["Y"], {"X": x})[0]
print(f"  输入: {x}")
print(f"  输出: {y}  (期望: x*2+1 = {x * 2 + 1})")
print()

# ─── 2. SessionOptions 详细配置 ───
print("=" * 60)
print("2. SessionOptions 配置示例")
print("=" * 60)

so = ort.SessionOptions()

# 线程控制
so.intra_op_num_threads = 2  # 单个算子内部并行线程数
so.inter_op_num_threads = 2  # 算子之间并行线程数

# 日志
so.log_severity_level = 3  # 0=Verbose 1=Info 2=Warning 3=Error 4=Fatal
so.log_verbosity_level = 0

# 内存
so.enable_cpu_mem_arena = False  # 关闭 CPU 内存池（调试时方便）
so.enable_mem_pattern = False

# 图优化
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 模型序列化（可选）
so.optimized_model_filepath = "optimized_model.onnx"  # 保存优化后的模型
sess2 = ort.InferenceSession("my_test_model.onnx", so)
y2 = sess2.run(["Y"], {"X": x})[0]
print(f"  输出: {y2}")
print()

# ─── 3. 查看三个图优化级别的速度差异 ───
print("=" * 60)
print("3. 图优化级别对比")
print("=" * 60)

N = 1000  # 推理轮次
x_big = np.random.randn(N, 4).astype(np.float32)

opt_levels = [
    # 不做任何优化 
    ("ORT_DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL), 
    # 做不影响图结构的优化，如：常量折叠（2.0 * 3.0 → 6.0）/死代码消除/冗余节点合并
    ("ORT_ENABLE_BASIC", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC), 
    # 包含 BASIC 的全部优化，做可能会改变图结构的优化，例如：算子融合/layout转换/子图替换 
    ("ORT_ENABLE_ALL", ort.GraphOptimizationLevel.ORT_ENABLE_ALL), 
]
for name, level in opt_levels:
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    so.log_severity_level = 3  # 只打 Error 日志，避免刷屏

    sess = ort.InferenceSession("my_test_model.onnx", so)

    start = time.perf_counter()
    for i in range(N):
        _ = sess.run(["Y"], {"X": x_big[i : i + 1]})
    elapsed = time.perf_counter() - start

    print(f"  {name:20s} | {N} 次推理耗时 {elapsed:.4f}s | 均值 {elapsed/N*1000:.3f}ms")

print()

# ─── 4. 查看可用 Provider ───
print("=" * 60)
print("4. 当前环境可用的 Provider")
print("=" * 60)

for p in ort.get_available_providers():
    print(f"  - {p}")
print()

# ─── 5. 模型元信息 ───
print("=" * 60)
print("5. 模型元信息")
print("=" * 60)

model_meta = sess2.get_modelmeta()
print(f"  生产者: {model_meta.producer_name}")
print(f"  图名:   {model_meta.graph_name}")
print(f"  域名:   {model_meta.domain}")
print(f"  自定义元数据: {model_meta.custom_metadata_map}")
print()

# ─── 6. 关掉优化的场景（调试自定义算子时有用） ───
print("=" * 60)
print("6. 关闭优化的调试模式（完整配置）")
print("=" * 60)

debug_so = ort.SessionOptions()
debug_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
debug_so.enable_cpu_mem_arena = False
debug_so.enable_mem_pattern = False
debug_so.log_severity_level = 0  # 打开详细日志
debug_so.intra_op_num_threads = 1
debug_so.inter_op_num_threads = 1

print("  SessionOptions:")
print(f"    优化级别:    {debug_so.graph_optimization_level}")
print(f"    CPU 内存池:  {debug_so.enable_cpu_mem_arena}")
print(
    f"    线程(内/间): {debug_so.intra_op_num_threads}/{debug_so.inter_op_num_threads}"
)
print(f"    日志级别:    {debug_so.log_severity_level}")
print()

# ─── 7. 验证结果一致性 ───
print("=" * 60)
print("7. 验证：开/关优化结果是否一致")
print("=" * 60)

x_test = np.random.randn(1, 4).astype(np.float32)

so_off = ort.SessionOptions()
so_off.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
so_off.log_severity_level = 3

so_on = ort.SessionOptions()
so_on.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so_on.log_severity_level = 3

sess_off = ort.InferenceSession("my_test_model.onnx", so_off)
sess_on = ort.InferenceSession("my_test_model.onnx", so_on)

y_off = sess_off.run(["Y"], {"X": x_test})[0]
y_on = sess_on.run(["Y"], {"X": x_test})[0]

diff = np.max(np.abs(y_off - y_on))
print(f"  随机输入: {x_test[0]}")
print(f"  关优化输出: {y_off[0]}")
print(f"  开优化输出: {y_on[0]}")
print(f"  最大差距:   {diff:.2e}")
print(f"  一致性:     {'✅' if diff < 1e-5 else '❌'}")
print()

print("Lesson 07 完成。")
