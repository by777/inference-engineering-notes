# Performance benchmarking and load testing · Software

<!-- https://learn-inference.com/chapters/software/benchmarking -->

Every optimization in this book is a hypothesis until you measure it. Benchmarking is the loop that turns configuration into knowledge, and most teams do it badly enough to draw wrong conclusions confidently.

### 4.5.1 Performance benchmarking tooling

There are good off-the-shelf harnesses: SGLang’s Genai-bench works against any engine, and NVIDIA’s GenAI-Perf runs client side. They generate load at a specified concurrency and report TTFT, inter-token latency, and throughput at percentiles.

Use one of these rather than writing your own. Getting concurrency and percentile accounting right is fiddly, and a subtly wrong harness produces numbers that look plausible and are not.

### 4.5.2 Performance benchmarking tips

Four things separate a useful benchmark from a misleading one.

-   **Use your real distribution.** Input and output lengths drive the prefill/decode ratio, which drives everything. A benchmark at 128 in and 128 out tells you nothing about a workload with 8,000-token prompts.
-   **Warm up first.** Cold caches, lazy kernel selection, and JIT compilation all make early requests slow. Discard them.
-   **Sweep concurrency.** A single number at one concurrency is nearly meaningless. The shape of the latency-versus-throughput curve is the actual result.
-   **Change one thing at a time.** Obvious, routinely ignored, and the reason so many teams cannot say which of their six changes helped.

### 4.5.3 Profiling performance

Benchmarking tells you the system is slow. Profiling tells you where. Nsight Systems gives you the timeline (gaps, serialization, synchronization stalls) and Nsight Compute goes inside a single kernel to say whether it is bound on memory or compute, which is [the roofline question](https://learn-inference.com/chapters/models/bottlenecks) asked of one kernel.

Profiling is not always the right tool. If you are running a configuration-driven engine like TensorRT-LLM, your loop is configure and benchmark, and a kernel profile is mostly interesting rather than actionable. Reach for it when you are writing kernels or when the benchmark is inexplicable.

That is the stack that gets a model running. Making it fast is a separate craft, a handful of techniques covered next.
