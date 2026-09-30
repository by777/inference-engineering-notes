# GPU architecture generations · Hardware

<!-- https://learn-inference.com/chapters/hardware/generations -->

Hardware moves slowly. Years separate a finalized architecture from shipping silicon, which means every GPU you can buy was designed against assumptions about AI workloads that predate the workloads you actually have. For most of that history the assumption was training. Only recently have architectures started shipping features aimed squarely at inference.

Names decompose cleanly. The letter is the architecture generation, named since 1998 for a scientist. The number identifies a model within it, and larger generally means bigger and more expensive. An H100 replaces an A100; an H200 is a larger card of the same generation; a B200 supersedes both.

> Figure. The same spec sheet, ranked three different ways. The current flagship leads on every axis, which is unsurprising. Watch second and third place instead: the H100 and H200 have identical FP8 compute, so they tie on prefill and the H200 pulls clearly ahead on decode. Same spec sheet, different answer, depending on what you are running.Constants from the book

### 3.2.1 Hopper GPUs

Hopper, for Rear Admiral Grace Hopper, arrived in 2022 with the H100. Its headline contribution to inference is FP8: tensor cores twice as fast as FP16 and values that cost half the bandwidth to move. As [Chapter 5](https://learn-inference.com/chapters/techniques/quantization) covers, that does not translate linearly into double the performance, but for workloads that tolerate it the gain is real.

Hopper also added asynchronous data transfer and execution, thread block clusters, and distributed shared memory. FlashAttention 3 is built directly on those primitives, a good example of hardware features only paying off once someone writes the kernel that exploits them.

The H100 and H200 remain among the most widely deployed inference accelerators, and the reason is boring and correct: new enough to be fast, old enough that every engine and kernel is tuned for them.

### 3.2.2 Ada Lovelace GPUs

Lovelace shipped six months after Hopper and is better understood as a counterpart than a successor. It supports FP8 too, but it is graphics-oriented, and critically it has no NVLink.

That omission matters more than the spec sheet suggests. Without a high-bandwidth interconnect these cards work alone or through inefficient pipeline parallelism, which rules out most of [the parallelism toolkit](https://learn-inference.com/chapters/techniques/parallelism). An L4 is a good, cheap way to serve embeddings or vision models. An L40 is usually the wrong answer for LLM serving: a fractional H100 gives you the same memory with much more compute and bandwidth.

### 3.2.3 Blackwell GPUs

Blackwell, for the mathematician David Blackwell, arrived in late 2024. Where Hopper introduced FP8, Blackwell pushes to FP4 along with a family of microscaling formats (MXFP8, MXFP4, NVFP4) designed to hold quality at precisions that would otherwise fall apart.

It extends Hopper’s asynchronous model with more machinery for moving data between tensor and global memory, which FlashAttention 4 leans on heavily. The B200 and B300 are the current high-water mark for large models and video generation, and the ecosystem has only recently caught up to them.

### 3.2.4 Rubin GPUs

Rubin, for the astronomer Vera Rubin, lands in 2026. Two details are worth knowing now. It moves to HBM4, which directly helps the bandwidth-bound decode phase. And it introduces the CPX, a separate chip built for compute-bound work like prefill. That is disaggregation, from [section 5.5](https://learn-inference.com/chapters/techniques/disaggregation), promoted from a software architecture into silicon.

Reserve judgement on any new architecture until you can benchmark your own workload on it. Announcement to broad software support runs about a year, and during that window the theoretical numbers and the achievable numbers are quite far apart. Feynman follows in 2028.

### 3.2.5 Grace and Vera CPUs

NVIDIA also builds ARM CPUs, paired with GPUs on superchips like the GH200 and GB200. For inference the interesting property is not CPU performance but the link between them: NVLink Chip-to-Chip runs up to 900 GB/s bidirectionally, several times what PCIe offers.

That changes what is worth offloading. Host memory is far larger than VRAM, so pushing LoRA weights or cold KV cache to the CPU is attractive in principle and often too slow in practice. At 900 GB/s it stops being a last resort. Vera replaces Grace alongside Rubin.
