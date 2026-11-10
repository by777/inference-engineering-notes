# CUDA · Software

<!-- https://learn-inference.com/chapters/software/cuda -->

CUDA is how you write code that runs on an NVIDIA GPU. It is described as a platform and a programming model, which is broad enough to be unhelpful; the concrete pieces are what matter.

-   A **kernel** (A function written to execute in parallel across many GPU threads.) is a function that executes in parallel across thousands of threads.
-   A **CUDA graph** is a directed acyclic graph of kernels and memory operations, captured once and replayed, which removes per-launch overhead.
-   **Streams** let independent work overlap instead of serializing.

### 4.1.1 CUDA kernels for inference

Almost nobody serving models writes kernels from scratch. The ones that matter (attention, normalization, the fused matmul-plus-activation paths) are written by specialists, published, and consumed through a library. FlashAttention is the canonical example: a kernel that changed what was economically possible, which almost every engine now depends on.

What you do need is the ability to read a profile and understand which kernel is slow and why. That is a different and much more achievable skill than writing one.

### 4.1.2 CUDA kernel selection

For any given operation there are usually several implementations, and which is fastest depends on shapes, precision, and architecture. A kernel tuned for a batch of 256 can be a poor choice at batch 1.

PyTorch and TensorRT-LLM both do automatic kernel selection, benchmarking candidates and caching the winner. This is why the first run of a compiled configuration is slow, and why the compiled artifact is worth caching across restarts.

### 4.1.3 Reducing memory accesses with kernel fusion

Running two kernels back to back on the same data wastes a round trip. The first writes its result to memory; the second immediately reads it back. The arithmetic is necessary; the extra round trip to memory is not.

> Figure 4.2. Kernel fusion, counted in round trips to memory. Two kernels back to back write an intermediate value out to HBM and immediately read it back. Fusing them deletes that round trip. Run both and compare the traffic counter: the arithmetic is identical, the memory traffic is not.Constants from the book

Real fusion is messier than `multiply_by_6`: functions are more complex and the data does not overlap so tidily. But the common patterns are well established, and combining a matrix multiplication with its bias add and activation is close to universal.

Note where this pays off. During [decode](https://learn-inference.com/chapters/models/bottlenecks), memory bandwidth is the binding constraint, so removing a round trip is removing time. During prefill, where you are compute-bound, the same fusion buys much less. Optimizations have a phase they belong to.
