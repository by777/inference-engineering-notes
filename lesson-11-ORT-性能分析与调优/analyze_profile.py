# analyze_profile.py
# Lesson 11: 分析 ORT profiling 输出的 json，打印热点算子耗时 Top N
# 用法: python3 analyze_profile.py <profile_file.json>

import json
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_profile.py <profile_file.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    # ORT profiling 输出格式（不同版本不同）:
    #   旧版: {"traceEvents": [...]}   (dict)
    #   新版 1.27+: 顶层直接是事件数组 (list)
    if isinstance(data, list):
        events = data
    else:
        events = data.get("traceEvents", [])
    print(f"total events: {len(events)}")

    # 按 op_name 聚合 kernel 耗时
    # 事件名形如 "fused Gemm1_kernel_time"，带节点名前缀，
    # 只有 args.op_name 才是算子类型（如 "FusedGemm"），用它聚合才有意义。
    kernel_time = defaultdict(float)
    for e in events:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            dur = e.get("dur", 0)  # 微秒
            args = e.get("args", {})
            node = args.get("op_name") or args.get("node_name") or "?"
            kernel_time[node] += dur

    print("\n===== Top 10 热点算子 (Kernel 执行时间, 微秒) =====")
    for name, total in sorted(kernel_time.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name:<30} {total:>12.0f} us")

    total_kernel = sum(kernel_time.values())
    print(f"\ntotal kernel time: {total_kernel:.0f} us")


if __name__ == "__main__":
    main()
