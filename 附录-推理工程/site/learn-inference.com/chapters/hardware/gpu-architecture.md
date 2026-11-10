# GPU architecture · Hardware

<!-- https://learn-inference.com/chapters/hardware/gpu-architecture -->

GPUs are throughput machines. A CPU is built to execute complicated sequential logic quickly; a GPU is built to perform one uniform operation across thousands of independent values at once. Since inference is a long chain of matrix multiplications, the fit is almost embarrassingly good: GPUs were designed for a different problem and happened to be exactly right for this one.

You will work several layers of abstraction above the silicon. But a decent mental model of what is inside the box is what lets you explain a benchmark result rather than just report it.

### 3.1.1 Compute

“Core” means something different here. A GPU is organized into **streaming multiprocessors** (Streaming Multiprocessor: the fundamental scheduling and execution unit of an NVIDIA GPU. A datacenter GPU has on the order of a hundred.), and each SM contains several kinds of execution unit:

-   **CUDA cores** operate on individual scalars.
-   **Tensor cores** operate on vectors and matrices. These do the actual work of inference.
-   **Special function units** accelerate transcendental operations like `sin`, `cos`, and `log`. Softmax depends on them.

When you compare GPUs for inference, compare tensor core throughput. Tensor cores execute matrix multiply-and-accumulate: multiply A by B, add C, store the result as D. That single instruction is the atom of everything in [Chapter 2](https://learn-inference.com/chapters/models).

Threads work roughly as you would expect from CPUs, except in quantity. A CPU juggles dozens to hundreds; a GPU runs tens or hundreds of thousands, switches between them in a single clock cycle, and executes them in lockstep groups.

Two spec-sheet traps

**Sparsity.** Published FLOPS often assume 2:4 structured sparsity, where half the values are zero and the tensor cores skip those multiplications. That roughly doubles the headline number. Inference is dense by default, so look for the dense figure.

**Precision.** FLOPS roughly double each time you halve precision. A card doing one petaFLOP at FP16 does about two at FP8. Comparing an FP8 number against an FP16 number makes one card look twice as good as it is.

### 3.1.2 Memory and caches

VRAM is the GPU’s onboard memory: high-bandwidth memory, HBM3 or HBM3e or HBM4 depending on generation, in the tens or hundreds of gigabytes. It is DRAM: large, off-chip, and slow relative to what sits above it.

Above it is SRAM, on-chip, measured in kilobytes and megabytes. L1 and shared memory live inside each SM; L2 is shared across all of them. An H100 carries 256 KB of L1 per SM and 50 MB of L2 in total.

> Figure. The memory hierarchy, and why the gaps matter. Both axes are logarithmic. Every step down is bigger and slower, and the ratio between the top two levels and HBM is what every serious kernel optimization is trying to exploit.Constants from the book

Two consequences follow, and they cover most hardware decisions you will make. **Capacity** caps the model you can load: weights plus at least fifty percent headroom for KV cache, more for long context or high batch sizes. Too little and the load fails outright with an out-of-memory error; too little headroom and inference is slow or crashes later, which is worse because it happens in production.

**Bandwidth** caps decode. At low to medium batch sizes you are reading the entire model out of VRAM for every token, so tokens per second tracks bandwidth almost directly. An H200 has identical compute to an H100 and roughly forty percent more bandwidth, which shows up almost entirely as more tokens per second, and barely at all in time-to-first-token.
