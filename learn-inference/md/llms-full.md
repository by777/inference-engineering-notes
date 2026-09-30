# Learn Inference: complete text

<!-- https://learn-inference.com -->

Every page of the book in reading order. Interactive figures appear as
labelled placeholders; open the linked page to use them.

---

# Learn Inference: inference engineering, explained interactively

<!-- https://learn-inference.com/ -->

# Learn Inference: an interactive guide to serving generative models in production

## Learn  
Inference

Serving generative  
models in production

[Start reading](https://learn-inference.com/chapters/inference)

[Continue reading](https://learn-inference.com/chapters/inference)

[Glossary](https://learn-inference.com/chapters/glossary)[Further reading](https://learn-inference.com/chapters/reading)[The source book](https://www.baseten.co/inference-engineering/)

## Everything after training

Training teaches a model what it knows. Inference is everything that happens afterward, every time somebody uses it, and it is where the bill actually lands. Serving a generative model well means working across a strange range of the stack: attention kernels at one end, GPU procurement across three clouds at the other.

This is an interactive companion to *Inference Engineering* by Philip Kiely. It follows the book’s structure and covers the same ground. I rewrote the explanations and built simulators for the parts you grasp faster by turning a dial than by reading a paragraph.

## Try one

This one is from chapter 1. Press run, then drag the sliders to feel the difference between starting fast and finishing fast.

> Figure 1.4. Time to first token versus tokens per second. Two numbers, two phases, two bottlenecks. A response that starts instantly and trickles can feel faster than one that pauses and then dumps, even when the second finishes first.Illustrative numbers

## Chapters

### [0\. Inference](https://learn-inference.com/chapters/inference)[](https://learn-inference.com/chapters/inference)

-   [0.1Two phases, two disciplines](https://learn-inference.com/chapters/inference/two-phases)
-   [0.2The three layers](https://learn-inference.com/chapters/inference/three-layers)
-   [0.3Six techniques that define the runtime](https://learn-inference.com/chapters/inference/runtime-techniques)
-   [0.4Scale changes the problem](https://learn-inference.com/chapters/inference/scale-changes-problem)
-   [0.5Where to put the abstraction](https://learn-inference.com/chapters/inference/abstraction)
-   [0.6A map of what follows](https://learn-inference.com/chapters/inference/map)

### [1\. Prerequisites](https://learn-inference.com/chapters/prerequisites)[](https://learn-inference.com/chapters/prerequisites)

-   [1.1Scale and specialization](https://learn-inference.com/chapters/prerequisites/scale-and-specialization)
-   [1.2About your app](https://learn-inference.com/chapters/prerequisites/about-your-app)
-   [1.3Model selection](https://learn-inference.com/chapters/prerequisites/model-selection)
-   [1.4Measuring latency and throughput](https://learn-inference.com/chapters/prerequisites/latency-throughput)

### [2\. Models](https://learn-inference.com/chapters/models)[](https://learn-inference.com/chapters/models)

-   [2.1Neural networks](https://learn-inference.com/chapters/models/neural-networks)
-   [2.2LLM inference mechanics](https://learn-inference.com/chapters/models/llm-mechanics)
-   [2.3Image generation inference mechanics](https://learn-inference.com/chapters/models/image-mechanics)
-   [2.4Calculating inference bottlenecks](https://learn-inference.com/chapters/models/bottlenecks)
-   [2.5Optimizing attention](https://learn-inference.com/chapters/models/optimizing-attention)

### [3\. Hardware](https://learn-inference.com/chapters/hardware)[](https://learn-inference.com/chapters/hardware)

-   [3.1GPU architecture](https://learn-inference.com/chapters/hardware/gpu-architecture)
-   [3.2GPU architecture generations](https://learn-inference.com/chapters/hardware/generations)
-   [3.3Instances](https://learn-inference.com/chapters/hardware/instances)
-   [3.4Other datacenter accelerator options](https://learn-inference.com/chapters/hardware/other-accelerators)
-   [3.5Local inference](https://learn-inference.com/chapters/hardware/local-inference)

### [4\. Software](https://learn-inference.com/chapters/software)[](https://learn-inference.com/chapters/software)

-   [4.1CUDA](https://learn-inference.com/chapters/software/cuda)
-   [4.2Deep learning frameworks and libraries](https://learn-inference.com/chapters/software/frameworks)
-   [4.3Inference engines](https://learn-inference.com/chapters/software/engines)
-   [4.4NVIDIA Dynamo](https://learn-inference.com/chapters/software/dynamo)
-   [4.5Performance benchmarking and load testing](https://learn-inference.com/chapters/software/benchmarking)

### [5\. Techniques](https://learn-inference.com/chapters/techniques)[](https://learn-inference.com/chapters/techniques)

-   [5.1Quantization](https://learn-inference.com/chapters/techniques/quantization)
-   [5.2Speculative decoding](https://learn-inference.com/chapters/techniques/speculative-decoding)
-   [5.3Caching](https://learn-inference.com/chapters/techniques/caching)
-   [5.4Model parallelism](https://learn-inference.com/chapters/techniques/parallelism)
-   [5.5Disaggregation](https://learn-inference.com/chapters/techniques/disaggregation)

### [6\. Modalities](https://learn-inference.com/chapters/modalities)[](https://learn-inference.com/chapters/modalities)

-   [6.1Vision language models](https://learn-inference.com/chapters/modalities/vlms)
-   [6.2Embedding models](https://learn-inference.com/chapters/modalities/embeddings)
-   [6.3ASR models](https://learn-inference.com/chapters/modalities/asr)
-   [6.4TTS models](https://learn-inference.com/chapters/modalities/tts)
-   [6.5Image generation models](https://learn-inference.com/chapters/modalities/image-models)
-   [6.6Video generation models](https://learn-inference.com/chapters/modalities/video-models)

### [7\. Production](https://learn-inference.com/chapters/production)[](https://learn-inference.com/chapters/production)

-   [7.1Containerization](https://learn-inference.com/chapters/production/containerization)
-   [7.2Autoscaling](https://learn-inference.com/chapters/production/autoscaling)
-   [7.3Multi-cloud capacity management](https://learn-inference.com/chapters/production/multi-cloud)
-   [7.4Testing and deployment](https://learn-inference.com/chapters/production/testing-deployment)
-   [7.5Client code](https://learn-inference.com/chapters/production/client-code)
-   [7.6Where this leaves you](https://learn-inference.com/chapters/production/closing)

### [Glossary](https://learn-inference.com/chapters/glossary)[](https://learn-inference.com/chapters/glossary)

Every term, defined once

### [Further reading](https://learn-inference.com/chapters/reading)[](https://learn-inference.com/chapters/reading)

-   [B.1Architecture](https://learn-inference.com/chapters/reading#architecture)
-   [B.2Developer tools](https://learn-inference.com/chapters/reading#developer-tools)
-   [B.3Frontier open models](https://learn-inference.com/chapters/reading#frontier-models)
-   [B.4GPU infrastructure](https://learn-inference.com/chapters/reading#gpu-infrastructure)
-   [B.5Inference optimization research](https://learn-inference.com/chapters/reading#optimization-research)
-   [B.6Intelligence evaluation](https://learn-inference.com/chapters/reading#evaluation)

---

# About

<!-- https://learn-inference.com/about -->

# About this site

Learn Inference is an interactive companion to *Inference Engineering* by Philip Kiely (Baseten Books, 2026). It follows the book’s structure and covers the same ground, with the explanations rewritten for the web and simulators built for the parts that are easier to understand by turning a dial than by reading a paragraph.

## Not affiliated

This is an independent project with no connection to Baseten or Philip Kiely. Nobody there reviewed it or signed off on it, and the rights to the book stay with its author and publisher. When a rewritten explanation or a figure here gets something wrong, the mistake was made here, not in the book.

The book itself is [available from Baseten](https://www.baseten.co/inference-engineering/).

## Who built it

The site does not say, but the Ask AI on any chapter will tell you if you ask.

## For developers

Learn Inference also publishes a keyless JSON API and an MCP server for the same chapter index. [See the developer page](https://learn-inference.com/developers).

---

# Glossary

<!-- https://learn-inference.com/chapters/glossary -->

Every term of art in this book, defined once. Search covers both the terms and their definitions, so the concept works even when the name does not come to mind.

## A

Activation function

—

A (mostly) differentiable nonlinear function like ReLU inserted between linear layers to prevent multi-layer neural networks from collapsing into a single matmul.

Active–active

—

A high-availability posture where multiple regions/clusters actively serve live traffic at once. If any plane fails, traffic seamlessly continues on the others.

Active–passive

—

A failover posture where a “hot standby” cluster or region is kept ready but idle. If the active plane fails, traffic is cut over to the passive plane.

Ada Lovelace (architecture)

—

NVIDIA’s graphics-oriented GPU architecture, released alongside Hopper in 2022. Useful for small models and cost-sensitive workloads, not suited for large-scale LLM inference.

Agent

—

An AI application that takes action rather than just providing information. Agent workflows usually rely on multiple inference calls, often across multiple models and modalities, and require access to tools.

AI-native application

—

A product where the core UX and value depend on generative models. Inference choices are downstream of app constraints: modality, latency budget, unit economics, and usage patterns.

Ampere (architecture)

—

An older NVIDIA GPU architecture still used in legacy or small-scale deployments. Hopper and Blackwell architectures generally outperform Ampere on both raw speed and cost at scale.

Application Programming Interface (API)

—

A structured interface for sending requests and receiving responses. Inference engines expose an API for making queries to models.

Arithmetic intensity

—

Operations performed per byte moved for a given algorithm. When compared to a GPU’s ops:byte ratio, arithmetic intensity indicates whether a kernel is compute bound or memory bound.

Attention

—

The core transformer mechanism relating a token to prior tokens via Q/K/V projections and softmax. Attention is a primary target for optimization due to its compute and memory demands.

Automatic Speech Recognition (ASR)

—

Audio-in, text-out transcription models (e.g., Whisper). Decoder work dominates runtime and benefits from LLM-style optimizations and in-flight batching.

Autoregressive token generation

—

Iterative generation of tokens where each token depends on each previous token. Autoregressive token generation is split into two phases: a prefill phase where input tokens are processed, and a decode phase where output tokens are generated.

Autoscaling

—

Scaling the number of replicas serving a given model up and down automatically based on traffic or utilization. Autoscaling matches capacity to demand, maintaining latency SLAs and minimizing wasted spend.

Autoscaling window

—

The rolling time horizon used to decide scale-up/ scale-down actions. Longer windows keep replica counts steady; shorter windows react faster to spikes.

## B

B200

—

NVIDIA Blackwell-based datacenter GPU with 192 GB of VRAM, 8 TB/s of memory bandwidth, and 5 petaFLOPS of FP8 compute.

B300

—

NVIDIA Blackwell-based datacenter GPU with 288 GB of VRAM, 8 TB/s of memory bandwidth, and 5 petaFLOPS of FP8 compute.

Bandwidth

—

The amount of data per second that can pass through memory like VRAM or an interconnect like NVLink.

Baselines

—

Initial, carefully recorded measurements of performance and quality before applying optimizations. Baselines enable clear attribution of gains or regressions.

Basic Linear Algebra Subprograms (BLAS)

—

A standard interface for fundamental operations in linear algebra.

Batch

—

Process multiple inputs simultaneously, common for LLM inference.

Batch sizing

—

A core latency-throughput lever for inference engines. Larger batches improve total throughput but worsen per-user latency.

Benchmark (intelligence)

—

A measurement of a model’s ability to answer questions correctly or take appropriate actions (e.g., MMLU).

Benchmark (performance)

—

A measurement of an inference service’s latency and throughput for a given model with a defined workload.

BF16

—

A 16-bit floating-point format with larger exponent than FP16, useful in training and sometimes inference. Higher dynamic range helps preserve outliers.

Bin packing (multi-cloud)

—

The practice of treating heterogeneous pools of GPUs across clouds, regions, and clusters as a single schedulable resource, enabled by multi-cloud capacity management infrastructure.

Blackwell (architecture)

—

NVIDIA’s late-2024 GPU generation featuring FP4 support, microscaling formats (MXFP8, MXFP4, NVFP4), and high memory bandwidth.

Blue-green deployment

—

Two parallel production environments (“blue” and “green”); shift traffic between them for zero-downtime deploys and quick rollback.

## C

Cache-aware routing

—

Steering requests to replicas that already hold matching prefixes or required LoRAs. Higher cache hit rates yield lower TTFT.

Canary deployment

—

A new production deployment that starts with a small share of live traffic to a new deployment to validate stability and performance. Over time, the new deployment absorbs all production traffic.

Causal language model (CLM)

—

A decoder-only transformer that predicts the next token given the prior context. All generative LLMs in this book are CLMs.

Central Processing Unit (CPU)

—

A general-purpose processor optimized for sequential workloads. CPUs are used for orchestration, scheduling, networking, and preprocessing, but rarely handle generative AI inference directly.

Chat template

—

The model-specific formatting and serialization of messages (roles, separators, beginning/end of sequence tokens).

Chunked prefill

—

Splits long inputs into chunks and overlaps prefill with decode or other work, preventing single long sequences from monopolizing resources.

Classifier-free guidance

—

Balances unconditional and prompt-conditioned denoising passes on each step of image generation. Lower guidance enhances creativity; higher guidance enforces prompt adherence.

CLIP (text encoder)

—

A text/image encoder used in earlier image pipelines (e.g., SDXL). Modern systems often swap in full LLMs for stronger prompt understanding.

Closed model

—

A proprietary model where weights are unavailable, like GPT-5, Claude Sonnet, or Google Gemini.

Cold start

—

The time from scaling a replica from zero to its first successful response (steps include GPU provisioning, container startup, model load, inference engine compilation).

ComfyUI

—

A workflow tool for assembling image pipelines (base model, refiner, LoRAs, ControlNets). Encourages modular, swappable components.

Compute-bound

—

An algorithm limited by available FLOPS rather than memory bandwidth. LLM prefill and image/video generation are usually compute bound.

Context Parallelism (CP)

—

Replicates weights across GPUs and partitions the attention context. Essential for video models where attention works across a massive latent space.

Context window

—

The maximum number of tokens that a model can process across input, reasoning, and output for a single request.

Continuous batching (in-flight)

—

Token-level interleaving of requests so GPU slots are always utilized. Minimizes per-user latency penalties of batching.

Control plane (multi-cloud)

—

Global orchestrator for deploying models and allocating resources.

Core (CUDA)

—

A general-purpose arithmetic unit that executes a wide range of scalar and element-wise operations.

Core (Tensor)

—

A specialized hardware unit optimized for mixed-precision matrix multiply-accumulate (MMA) operations. Tensor Cores are the most important type of compute for inference.

Cross-attention

—

Conditioning one sequence (Q) on another’s K/V (e.g., text conditioning images). Common in multimodal and denoising pipelines.

cuBLAS

—

CUDA’s BLAS implementation offering high-quality GEMM and related primitives.

CUDA

—

NVIDIA’s programming model and platform for GPU kernels, graphs, memory, and execution.

CUDA driver

—

A low-level interface between the application and the GPU hardware to manage memory and execution.

CUDA graph

—

A directed acyclic graph (DAG) of kernels and other GPU operations for optimizing repeated workflows.

CUDA kernel

—

A user-defined function that executes parallelized code on the GPU.

CUDA runtime

—

A developer-facing API for launching kernels and managing memory.

cuDNN

—

Primitives for building deep neural networks in CUDA.

CuTe

—

A domain-specific C++ template library that abstracts tiled tensor operations to help developers compose precision-aware, hardware-optimized GEMM and fused kernels.

CUTLASS

—

A CUDA C++ template library that provides building blocks for writing high-performance, architecture-tuned GEMM and related kernels.

## D

Data sovereignty

—

Legal constraints around where model inputs and outputs are processed and stored geographically.

Decode

—

The memory-bound phase of LLM inference where the autoregressive generation loop emits one token per forward pass.

DeepGEMM

—

A library of clean and efficient GEMM kernels created by the DeepSeek AI team with strong performance in FP8.

Denoising model

—

The heart of diffusion pipelines that iteratively refines latent noise into an image or video.

Diarization

—

Segmenting audio by speaker (“who spoke when”); often paired with VAD in ASR pipelines.

Diffusers (library)

—

Reference implementations for image and video generation pipelines.

Disaggregation

—

Separating prefill and decode onto independently scaling engines running on separate hardware resources.

Distillation

—

Training a smaller student to emulate a larger teacher model based on probability distributions, not just outputs, to retain model behavior on fewer parameters.

Docker

—

Containerization technology for building standardized packages of inference services with their dependencies.

Dockerfile

—

A human-readable file with well-specified, machine-interpretable instructions for creating an image.

Dynamic batching

—

Dynamic batching starts a batch when the batch is full or a short timer elapses, whichever comes first. Balances latency stability with utilization; superseded by continuous batching for LLMs.

Dynamic range (quantization)

—

The range of absolute values that can be represented in a number format. Floating-point numbers have a higher dynamic range than integers with the same number of bytes thanks to their exponent-mantissa structure.

## E

EAGLE (speculation)

—

A small, purpose-built draft model trained to consume hidden states and propose multiple tokens for high acceptance rates in speculative decoding.

Elo (quality meta-metric)

—

Head-to-head win-rate style scoring to compare model quality. Useful directional signal beyond intelligence benchmarks.

Embedding model

—

Encodes text or image input into fixed-dimensional vectors for semantic similarity, used in RAG and agent memory. Modern variants often use LLM backbones and Matryoshka representations.

Encoder

—

Network that converts raw inputs into internal representations (e.g., audio features in Whisper). Paired with a decoder in encoder-decoder models.

Evals

—

Task-specific tests that mirror real-world use cases for a model, used for product-specific model intelligence testing.

Expert Parallelism (EP)

—

Shards experts of an MoE across GPUs; each GPU contains multiple full experts. Increases total throughput with low inter-GPU communication overhead.

## F

Few-step image generation

—

Models that produce usable images in eight or fewer steps. Eighty to ninety percent faster but with noticeable quality tradeoffs; strong fit for real-time applications.

Feynman (architecture)

—

A future NVIDIA generation after Rubin. Details are limited; expect continued emphasis on low-precision and memory bandwidth.

Fine-tuning

—

Adapts a pretrained base to a domain, often enabling much smaller models to meet quality needs.

FlashAttention

—

A series of optimized attention kernels that minimize memory traffic. FlashAttention 3 is written for Hopper, FlashAttention 4 targets Blackwell.

Floating-point data formats

—

Precisions like FP16, FP8, and FP4 used in inference with high dynamic range and an exponent-mantissa structure.

FLOPS

—

Floating-point operations per second, typically measured on Tensor Cores.

Foundation model

—

A model trained on broad data that serves as a base for multiple downstream tasks. Foundation models (e.g., GPT, Claude, Llama) are typically fine-tuned or used directly via prompting.

Function calling

—

Also known as tool calling or tool use, a model is given a set of available functions along with a prompt and returns a structured output including both selected functions and arguments for those functions.

## G

GB200

—

An NVIDIA superchip that pairs a Grace CPU with a B200 GPU via high-bandwidth NVLink chip-to-chip connection. GB200s are used in rackscale NVLink systems like the NVL72 and are useful for KV cache offloading, LoRA swapping, and other techniques that benefit from NVLink-C2C.

General matrix-matrix multiplication (GEMM)

—

An algorithm in BLAS and the key operation for inference.

Generative AI

—

A class of models that, in contrast to predictive ML models, create new content across modalities (text, images, audio, video, code) by learning the underlying patterns of training data.

Generative Pretrained Transformer (GPT)

—

A family of large language models for text generation created by OpenAI.

GH200

—

An NVIDIA superchip that pairs a Grace CPU with an H200 GPU via high-bandwidth NVLink chip-to-chip connection. GH200s are used in rackscale NVLink systems like the NVL72 and are useful for KV cache offloading, LoRA swapping, and other techniques that benefit from NVLink-C2C.

Goodhart’s Law

—

“When a measure becomes a target, it ceases to be a good measure.”

GPU node

—

A standard chassis of 8 interconnected GPUs with NVLink and NVSwitch.

Grace CPU

—

ARM-based NVIDIA CPU with high-bandwidth chip-to-chip interconnects between the CPU and GPU. Used alongside Hopper and Blackwell GPUs.

Graphics Processing Unit (GPU)

—

A highly parallel processor originally designed for graphics rendering and now widely used for training and inference of generative AI models.

gRPC

—

Structured, schema-first bidirectional streaming protocol.

## H

Head (attention)

—

One independent attention computation within a layer.

High-Bandwidth Memory (HBM)

—

The memory used for VRAM on datacenter GPUs. Recent generations include HBM3, HBM3e, and HBM4.

Hopper (architecture)

—

NVIDIA’s 2022 GPU generation featuring FP8 support and async programming features.

Hyperscaler

—

Generalized cloud service providers like AWS and GCP.

## I

Image generation pipeline

—

Foundation models for image generations are pipelines of multiple models: a text encoder, an iterative denoiser, and a VAE.

In-flight batching

—

See continuous batching. Token-level interleaving for high utilization with stable latency.

Inference

—

Serving AI models in production.

Inference engine

—

A high-performance runtime (vLLM, SGLang, TensorRT-LLM) with support for optimization techniques like batching, caching, quantization, and speculation.

InfiniBand

—

Inter-node interconnect for scaling inference and training across multiple nodes. While InfiniBand bandwidth is higher than alternatives like Ethernet, it is substantially lower than NVLink.

Input sequence

—

The tokens provided to a model as part of a request, processed during the prefill phase of inference.

Input Sequence Length (ISL)

—

The number of tokens in the input sequence for a given request.

Instance (cloud)

—

The provisioned virtual machine that includes GPU(s), CPU and RAM resources, storage, networking, and interconnect.

Integer data formats

—

Number formats like INT8 and INT4 with limited dynamic range.

Inter-token latency (ITL)

—

Time between generated tokens during decode. Converts to perceived TPS (e.g., 2 milliseconds ITL equates to 500 TPS).

Iterative denoising (diffusion)

—

Start from noise and progressively refine into an image or video in latent space.

## J

Jitter traffic (bench)

—

Adding randomness to arrival times and sequence shapes to more closely mirror real traffic than uniform or bursty synthetic loads.

## K

Kernel fusion

—

Taking two or more kernels and re-implementing them into a single kernel that handles both operations, avoiding unnecessary round-trips through memory.

KV cache

—

Stored K/V tensors for each token to avoid recomputing attention, turning the attention equation from a quadratic-time to a linear-time operation.

## L

L0/L1/L2 caches (GPU)

—

On-chip cache memory hierarchy for instructions, shared memory, and global cache.

Large Language Model (LLM)

—

A type of generative AI model that takes a text prompt and returns a new sequence of text. Many famous generative AI model families, including GPT, Claude, Llama, and DeepSeek, are LLMs.

Latency percentiles

—

Measuring latency on a percentile basis (P50/P90/ P95/P99) for awareness of both the average and the worst-case user experience.

Latent consistency

—

A few-step strategy that predicts target latents directly, possibly repeated for refinement. Very fast; lower fidelity than full diffusion.

Latent space (images/videos)

—

Lower-dimensional representation where denoising occurs (e.g., 128×128).

LLM

—

Large language model (e.g., GPT-5, Llama, DeepSeek).

Load testing

—

Sending sustained high traffic to probe throughput limits, queue behavior, and autoscaling.

Local (edge) inference

—

Running inference on end-user devices like phones and computers.

Logit biasing

—

Nudging or constraining token probabilities to steer structured outputs (e.g., JSON/tool calls). Applied post-logits before sampling.

Logits

—

A vector of non-normalized probabilities, one per token in the model’s vocabulary, generated in each forward pass during decode.

Lookahead decoding

—

Constructs n-grams during inference to enable draft token prediction without a separate model.

LoRA

—

Low-rank adaptation, a lightweight fine-tuning method that produces small changes to models. Inference services often need to swap between thousands of LoRAs for a single foundation model.

## M

Machine learning (ML)

—

Predictive modeling for tasks like classification and trend forecasting, as opposed to generative AI which creates novel outputs.

Matmul

—

Matrix multiplication.

Matryoshka representations (embeddings)

—

Nested vector schemes allowing variable dimensionality where the early part of the vector encodes more semantic meaning. Allows tradeoffs between vector size and embedding quality.

Medusa (speculation)

—

Adds extra decoder heads via fine-tuning to generate multiple draft tokens per pass.

Microscaling formats

—

Floating-point data formats like MXFP8, MXFP4, and NVFP4 that use blockwise quantization with small-block scale factors (e.g., every 32 elements) to improve accuracy.

Mixture of Experts (MoE)

—

A model architecture where linear layers of weights are separated into sparse experts. A router activates a subset of experts for each forward pass.

Model parallelism (overview)

—

Splitting work across GPUs via Tensor, Expert, or Pipeline Parallelism. Parallelism strategy depends on model size, topology, and latency versus throughput goals.

Multi-cloud capacity management

—

A global scheduler placing workloads across providers and regions.

Multi-Instance GPU (MIG)

—

A capability in larger Ampere, Hopper, Blackwell, and Rubin GPUs where the GPU can be carved into up to eight slices of memory and seven slices of compute.

Multi-node inference

—

Scaling across two or more nodes using InfiniBand when one node of eight GPUs doesn’t have enough VRAM for weights, activations, and KV cache. Requires appropriate parallelism strategies: PP or EP between nodes, as TP uses too much all-to-all communication for InfiniBand.

## N

N-gram speculation

—

Uses observed n-grams from prefill to propose long draft sequences during decode. Extremely effective for code completion.

Neocloud

—

Specialized cloud service providers focused on GPUs like Coreweave and Nebius.

Neural audio codec

—

A learned encoder that compresses audio into tokens and paired decoder that turns tokens back into audio.

NIM

—

A pre-packaged, containerized microservice for a specific model created by NVIDIA.

Node

—

The physical 8-GPU base unit with NVLink/NVSwitch. Multi-node adds InfiniBand between nodes.

NVFP4

—

NVIDIA’s 4-bit floating-point microscaling number format with dual scale factors and blockwise quantization with a block size of 16.

NVIDIA Dynamo

—

An open-source distributed serving platform for KV reuse, disaggregation, and multi-GPU/multi-node orchestration.

NVL72

—

A rack-scale Blackwell system interconnecting 72 GPUs and 36 CPUs. Purpose-built for serving very large models with extreme throughput.

NVLink

—

A one-to-one communication layer between GPUs, up to 1800 GB/s on Blackwell and 900 GB/s on Hopper.

NVSwitch

—

An all-to-all communication layer on top of NVLink for coordination among all GPUs in a node.

## O

Offline inference

—

Asynchronous batch processing of large jobs, optimized for throughput and cost over per-request latency.

Omni-modal

—

Models that accept multiple modalities of inputs (text, images, video, audio) and produce multiple modalities of output.

Online inference

—

Real-time serving of requests, optimized for tight latency budgets.

ONNX

—

An intermediate representation and runtime for models.

Open model

—

A model whose weights are freely available, like Llama, DeepSeek, or Whisper.

Ops:byte ratio (GPU)

—

Peak operations per byte of memory bandwidth for a GPU at a given precision. Compare with arithmetic intensity to diagnose bottlenecks.

Out-of-memory error (OOM)

—

A common failure where the GPU runs out of VRAM to load weights or execute inference.

Output sequence

—

The tokens generated by a model during the decode phase of inference.

Output Sequence Length (OSL)

—

The number of tokens in the output sequence generated by a model for a given request.

## P

PagedAttention

—

An optimization for attention where KV blocks are stored in fixed-size pages to improve performance, especially with long context.

PCIe (GPU form factor)

—

A form factor for datacenter GPUs that uses standard PCI express slots for connection. PCIe GPUs often have lower base specs and fewer interconnect options than SXM variants of the same GPU.

Perceived TPS

—

Tokens per second observed by a single user during streaming output. This latency metric is a more specific term for what people usually mean when they say TPS.

Pipeline Parallelism (PP)

—

Splits layers into stages across GPUs. While acceptable for multi-node with dense models; PP introduces bubbles in the pipeline where some GPUs are idle while waiting for other steps to finish.

Prefill

—

The compute-bound phase of LLM inference where the input sequence is processed and the KV cache is built.

Prefix caching

—

Reuses KV for shared prefixes across requests to skip prefill. Majorly improves TTFT for code completion, multi-turn chat, and agents.

Pretraining

—

Large-scale (usually self-supervised) training on broad corpora to create a base model.

Prompt

—

The instruction to the model; for diffusion also includes a negative prompt and step/guidance parameters.

PyTorch compile (torch.compile)

—

Graph capture and kernel selection/fusion targeting a specific GPU. Cache compiled engines to cut cold-start times.

PyTorch Profiler

—

A developer tool measuring CPU and GPU time and memory per operation.

## Q

Quantization (post-training)

—

Lowering precision of weights, activations, and potentially KV cache to reduce compute and memory bandwidth demands.

Quantization-aware training

—

A training technique in which quantization scales are computed and weights are optimized jointly so that the final model is already calibrated for low-precision deployment.

Queue (request)

—

Holds excess traffic while autoscaling brings replicas online.

## R

Real-time factor (RTF)

—

A measurement of how quickly ASR models can transcribe audio. Transcribing an hour of audio in six seconds is an RTF of 600.

Retrieval-augmented generation (RAG)

—

A common application pattern that fetches additional context for the LLM beyond the prompt.

Ring attention

—

A Context Parallelism mechanism in which GPUs pass partial attention results in a ring. Reduces all-to-all pressure for very large contexts.

Roofline model

—

Plots arithmetic intensity with bandwidth and compute ceilings, creating a visual guide on whether to optimize memory or compute.

Rotary positional embeddings (RoPE)

—

A positional encoding scheme that encodes positions as learned rotations, improving long-context extrapolation at the cost of higher memory demands for attention during inference.

Routing (inference)

—

Placing requests on replicas based on load, KV cache, available LoRAs, and sequence shapes to improve speed and utilization.

Rubin (architecture)

—

Next NVIDIA generation (2026) introducing HBM4 and CPX for compute-bound workloads.

## S

Sampling (decode)

—

The process of selecting an output token based on the generated logits. Common strategies include greedy (argmax), temperature-based sampling, top-k, and top-p (nucleus) sampling.

Scale factor (quantization)

—

Multipliers used to map low-precision values to their original number formats.

Scale to zero

—

Turn off all replicas when idle; spin up on demand. Requires fast cold starts and robust queueing; best for predictable or dev workloads.

SDXL

—

An instructive, earlier diffusion image pipeline (base + refiner + CLIP). Modern systems retain the structure with larger, more capable components.

Service Level Agreement (SLA)

—

A contractual promise of latency, throughput, uptime, or other performance factor from a system.

Service Level Objective (SLO)

—

An internal target designed to meet or beat the SLA for a given system.

SGLang

—

A fast inference engine with flexible frontend/backends and strong MoE support.

Shadow traffic

—

Mirroring real production requests to a candidate deployment.

SNAC (audio decoder)

—

A performant audio decoder path often paired with TTS token streams.

Softmax

—

Converts scores to probabilities in attention and normalizes logits to a probability distribution in decoding.

Sparsity (FLOPS)

—

In tensors with 2:4 structured sparsity, where 50 percent of the values are 0, Tensor Cores can skip multiplication by 0. Most inference is dense, not sparse.

Special Function Unit (SFU)

—

A dedicated hardware unit that accelerates specific math operations like sine and cosine, keeping specialized operations off of CUDA Cores.

Speculative decoding

—

A family of strategies for generating and validating draft tokens to generate multiple tokens per forward pass during decode.

Streaming Multiprocessor (SM)

—

GPU compute unit containing cores and cache.

Structured output

—

LLM output that adheres to a specific schema. Created by constraining generation to a supplied schema via logit biasing rather than via prompting.

SXM (GPU form factor)

—

A socketed GPU module that supports higher-bandwidth connections and delivers more power than PCIe. SXM form factor GPUs often have higher base specs and are the standard for inference.

## T

Temperature

—

Controls randomness in token selection: lower values (e.g., 0.1) make output more deterministic; higher values (e.g., 1.5) increase diversity.

Tensor Parallelism (TP)

—

Splits tensor operations across GPUs within a node. Best per-user latency; requires frequent all-reduce synchronization.

TensorRT

—

NVIDIA’s optimized runtime for high-performance inference with fused kernels, quantization, and other optimizations.

TensorRT-LLM

—

An inference engine built by NVIDIA that provides a Python API and both TensorRT-engine and PyTorch-backend execution paths with fused kernels, quantization, and speculative decoding.

Thread

—

The minimal execution unit on a GPU. Kernels launch many threads to achieve massive parallelism.

Throughput

—

Total work per unit time (e.g., total tokens per second).

Time to first byte (TTFB)

—

Time until first byte of output is returned, a latency metric.

Time to first token (TTFT)

—

Time until first token of output is returned, a latency metric.

Token

—

The atomic unit of text processing in LLMs. A token is an integer that represents a string of characters. In English, there is approximately a 4:3 token:word ratio for most tokenizers.

Tokenizer

—

Deterministically converts strings into sequences of tokens, and vice versa. Models have different tokenizers, and more efficient tokenizers improve end-to-end latency.

Tokens per second (TPS)

—

See perceived TPS. A latency metric for the number of tokens streamed to the end user per second.

Training

—

The process of learning model weights from data using backpropagation and optimization. Training is compute-intensive, typically runs on large GPU clusters, and produces the weights used in inference.

Transformer

—

The foundational architecture behind generative AI models.

Transformers (library)

—

Reference implementations for LLMs and other transformers-based models.

Triton Inference Server

—

A production serving framework by NVIDIA with support for multiple backends.

## V

VAE (variational autoencoder)

—

Used in inference to decode from latent space to pixel space for image and video generation models (also used for encoding from pixel to latent space during training).

Vector database

—

A database for storing and querying the semantic vectors created by embedding models.

Vector similarity

—

A check between two vectors to see how close together they are based on an equation like cosine similarity. Vectors with high similarity encode similar semantic meaning.

Vera CPU

—

ARM-based NVIDIA CPU with high-bandwidth chip-to-chip interconnects between the CPU and GPU. Succeeds Grace GPUs alongside the Rubin GPU architecture generation.

Vision-language model (VLM)

—

Accepts images and video plus text prompts and outputs text.

vLLM

—

A widely adopted inference engine with broad model and hardware support and strong defaults.

Vocabulary

—

The total set of tokens, usually more than 100,000, that an LLM uses to represent data.

Voice activity detection (VAD)

—

A lightweight model that segments streams/files into speech-containing chunks for ASR.

VRAM (device memory)

—

On-GPU memory used for weights, KV, and activations. Total VRAM gates model size and KV headroom; bandwidth gates decode TPS.

## W

WebSocket

—

Lightweight, bidirectional streaming transport. Ideal for unstructured audio chunks and real-time UX.

Weights-only quantization

—

Reduces precision for model weights in linear layers while preserving other model components like KV cache and attention at higher precision. A conservative approach to quantization with the best quality preservation but the lowest performance improvements.

Workload plane (multi-cloud)

—

An individual cluster with compute resources that runs inference and processes requests.

On these definitions

Unlike the rest of this site, the glossary quotes *Inference Engineering* directly. A definition of a term of art is a factual statement, and paraphrasing one for the sake of paraphrasing makes it worse rather than more original.

---

# Hardware

<!-- https://learn-inference.com/chapters/hardware -->

A spec sheet has three numbers on it that matter, and which one you should care about depends entirely on what you are running. Reading them correctly is most of what hardware selection is.

Inference runs on accelerators, and in the datacenter that overwhelmingly means NVIDIA GPUs. There are three broad classes: datacenter GPUs racked in standardized chassis, workstation GPUs for professional desktops, and consumer cards. Serving at scale means the first category, not because the silicon is categorically better but because those cards ship with high-bandwidth interconnects, arrive in predictable configurations, and exist by the million.

You are almost certainly renting them. Cloud is the default: hyperscalers like AWS and GCP, or neoclouds like CoreWeave and Nebius. On-premise and air-gapped deployments are real but belong to large enterprises and governments; everyone else wants the ability to add capacity without a procurement cycle.

## 5 sections

1.  [3.1GPU architectureCompute · Memory and caches](https://learn-inference.com/chapters/hardware/gpu-architecture)
2.  [3.2GPU architecture generationsHopper GPUs · Ada Lovelace GPUs · Blackwell GPUs · Rubin GPUs · Grace and Vera CPUs](https://learn-inference.com/chapters/hardware/generations)
3.  [3.3InstancesMulti-GPU instances · Multi-instance GPUs](https://learn-inference.com/chapters/hardware/instances)
4.  [3.4Other datacenter accelerator options](https://learn-inference.com/chapters/hardware/other-accelerators)
5.  [3.5Local inferenceDesktop inference · Mobile inference](https://learn-inference.com/chapters/hardware/local-inference)

---

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

---

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

---

# Instances · Hardware

<!-- https://learn-inference.com/chapters/hardware/instances -->

You do not rent a GPU. You rent an instance: a virtual machine that happens to include one or more GPUs, plus host CPUs, host memory, storage, networking, and interconnect. Every one of those can be the thing that ruins your latency.

This is where cloud providers diverge. NVIDIA publishes reference architectures; providers build to their own preferences. Even the same GPU differs: an A100 comes in PCIe and SXM form factors, and SXM has about five percent higher memory bandwidth, enough that most A100 inference runs on it. Know exactly what you are provisioning, because a slow disk or a thin network link will bottleneck a perfectly good GPU.

### 3.3.1 Multi-GPU instances

The standard unit is a node of eight GPUs, connected by **NVLink** (NVIDIA's direct GPU-to-GPU interconnect. Up to 900 GB/s on Hopper and 1,800 GB/s on Blackwell.) for point-to-point traffic and NVSwitch for all-to-all coordination. Those links are what make it practical to split one model across eight cards.

Past eight, you need node-to-node interconnect, and the standard is InfiniBand at up to 400 Gb/s per NIC. Note the units: NVLink is measured in gigabytes per second, InfiniBand in gigabits. NVLink is roughly an order of magnitude faster.

That ratio governs every parallelism decision in [Chapter 5](https://learn-inference.com/chapters/techniques/parallelism). A strategy that saturates InfiniBand may be comfortable inside a node. Not every provider offers InfiniBand, and some offer it on only part of their fleet, so check before you design around it.

At the top end, NVIDIA sells rack-scale systems: the GB200 NVL72 puts 72 Blackwell GPUs and 36 Grace CPUs under one NVLink domain, with the Vera Rubin NVL144 CPX to follow. These exist to serve the largest models at serious traffic.

### 3.3.2 Multi-instance GPUs

The opposite problem is more common than people expect: the GPU is too big for the model. A three-billion parameter model cannot keep an H100 busy even at generous batch sizes, and the waste is pure margin loss.

Multi-instance GPU splits an A100, H100, H200, or B200 into as many as seven slices, each with its own compute, memory, and a proportional share of CPU, RAM, storage, and network. An H100 has eight memory slices and seven compute slices, which is why the partitioning looks strange to anyone expecting powers of two.

This is usually the better answer than reaching for an older, cheaper card. A fraction of a modern GPU carries modern architecture features and modern kernel support; an equivalently priced older card does not.

---

# Local inference · Hardware

<!-- https://learn-inference.com/chapters/hardware/local-inference -->

Not all inference happens in a datacenter, and the constraints invert when it does not. There is no batching across users, because there is one user. Memory is small and shared with everything else on the device. Power and thermals matter in a way they never do on a rack.

### 3.5.1 Desktop inference

Consumer cards and Apple Silicon both run useful models now. The interesting property of unified memory architectures is that the usual VRAM ceiling does not apply the same way: the constraint becomes bandwidth rather than capacity, which changes which models are actually pleasant to run.

Batch size one is the defining fact. Every arithmetic-intensity argument from [Chapter 2](https://learn-inference.com/chapters/models/bottlenecks) applies at its most extreme, so local inference is memory-bandwidth-bound almost by definition, and quantization stops being an optimization and becomes a precondition.

### 3.5.2 Mobile inference

Phones ship neural accelerators (Apple’s Neural Engine, Qualcomm’s Hexagon) alongside their CPUs and GPUs. The models are small, heavily quantized, and usually converted to a device-specific runtime rather than served from PyTorch.

What makes on-device worth the trouble is rarely speed. It is that the data never leaves the device, and that the feature works on a plane. Those are product arguments, and they are often decisive regardless of what the benchmarks say.

Hardware sets the ceiling on all of this. How close you get to it is a question about the software that drives the chip, which is where the book turns next.

---

# Other datacenter accelerator options · Hardware

<!-- https://learn-inference.com/chapters/hardware/other-accelerators -->

NVIDIA’s dominance in inference is real but not total. AMD ships the MI350 with competitive specs on its own software stack, which has improved to the point where major engines run on it. Google’s TPUs serve enormous internal and Cloud workloads. AWS builds Inferentia and Trainium. Groq and Cerebras take different architectural approaches aimed at latency.

The gap is rarely the silicon. It is CUDA and the decade of kernels, libraries, and engine support built on top of it. Moving off NVIDIA means accepting that some optimization you want will not be available yet, and pricing that against whatever capacity or cost advantage prompted the move.

---

# Inference

<!-- https://learn-inference.com/chapters/inference -->

A generative model is trained once and then serves for the rest of its life. This chapter is about that second phase: what serving actually involves, and the three layers of engineering it takes to do well.

## 6 sections

1.  [0.1Two phases, two disciplines](https://learn-inference.com/chapters/inference/two-phases)
2.  [0.2The three layers](https://learn-inference.com/chapters/inference/three-layers)
3.  [0.3Six techniques that define the runtime](https://learn-inference.com/chapters/inference/runtime-techniques)
4.  [0.4Scale changes the problem](https://learn-inference.com/chapters/inference/scale-changes-problem)
5.  [0.5Where to put the abstraction](https://learn-inference.com/chapters/inference/abstraction)
6.  [0.6A map of what follows](https://learn-inference.com/chapters/inference/map)

---

# Where to put the abstraction · Inference

<!-- https://learn-inference.com/chapters/inference/abstraction -->

Once the runtime and infrastructure work exists, somebody has to decide how it is exposed. This applies whether you are buying inference from a provider or building a platform for your own engineers; the question is the same either way.

At one extreme, inference is a black box: hand over weights, receive an API. At the other, you get compute, network, and disk, and everything above that is yours. Both extremes are defensible and most teams belong somewhere between them.

> Figure. How much of the stack do you want to own?. Control and productivity trade against each other along one axis. Each step down buys the ability to tune deeper layers and costs the engineering time to run them. Take as much abstraction as your requirements allow, and give up productivity only where something you need demands the control.

I would push most teams further toward abstraction than instinct suggests. Owning the whole stack is satisfying and it is occasionally correct, but it is a standing commitment of engineering time that competes directly with the product. Keep only as much control as you have a concrete use for.

---

# A map of what follows · Inference

<!-- https://learn-inference.com/chapters/inference/map -->

The rest of the book works up the stack. Each chapter assumes the previous ones but stands on its own well enough to be read out of order if you came here for something specific.

-   **[Chapter 1, Prerequisites](https://learn-inference.com/chapters/prerequisites):** Every inference optimization is an answer to a question about your product. Before touching a kernel, you need to know what you are building, which model you are serving, and what 'fast enough' means in numbers rather than adjectives.
-   **[Chapter 2, Models](https://learn-inference.com/chapters/models):** At serving time, a model is a long chain of matrix multiplications. This chapter builds up from a linear layer to a transformer block to a diffusion pipeline, then does the arithmetic that tells you whether your bottleneck is memory bandwidth or compute: the calculation worth doing once to build the intuition.
-   **[Chapter 3, Hardware](https://learn-inference.com/chapters/hardware):** Compute, memory, and the bandwidth between them. This chapter takes apart a modern datacenter GPU, disambiguates NVIDIA's architecture names and SKUs, and covers what changes when a model no longer fits on one card.
-   **[Chapter 4, Software](https://learn-inference.com/chapters/software):** The stack between a model file and a served token: CUDA, PyTorch, the model formats, and the three inference engines that most production deployments actually run on: vLLM, SGLang, and TensorRT-LLM.
-   **[Chapter 5, Techniques](https://learn-inference.com/chapters/techniques):** Quantization, speculative decoding, KV cache re-use, model parallelism, and disaggregation. Each one trades precision, memory, complexity, or hardware for latency or throughput. This chapter is about knowing which trade you are making.
-   **[Chapter 6, Modalities](https://learn-inference.com/chapters/modalities):** Most non-text models are transformer variants, which means the LLM toolkit mostly transfers. Image and video generation are the exception: different architecture, different bottlenecks, different optimizations.
-   **[Chapter 7, Production](https://learn-inference.com/chapters/production):** Containers, autoscaling, cold starts, multi-cloud capacity, zero-downtime deploys, and the client code on the other end. A perfectly optimized runtime still falls over if the infrastructure around it cannot get GPUs when traffic arrives.

Two appendices close it out: a [glossary](https://learn-inference.com/chapters/glossary) of the vocabulary, and a set of [primary sources](https://learn-inference.com/chapters/reading) worth reading directly.

On knowledge cutoffs

Books have them too. The source material was finished in January 2026, and specific hardware, version numbers, and benchmark figures will drift. The underlying constraints move much more slowly. Memory bandwidth is the binding limit on decode today, and nothing about that is about to change.

---

# Six techniques that define the runtime · Inference

<!-- https://learn-inference.com/chapters/inference/runtime-techniques -->

The runtime layer is largely an exercise in applying a handful of techniques, most of them adapted from research published in the last few years. They come up constantly, they compose with each other, and each gets a proper treatment later in the book. It is worth meeting them now, because the rest of the material is organized around them.

What is easy to miss on a first pass is that none of these is free. Every one takes something away in exchange for what it gives, and choosing well means knowing which currency you are spending.

> Figure. Every technique is a purchase. Toggle techniques on the way you would configure a deployment, and the bill fills in: one line per cost, filed under the currency it spends. Turn everything on and count the rows: complexity is billed by four of the six techniques, and quality exactly once, by quantization. No combination of switches leaves the bill empty.

These apply well beyond text. Vision language models, embedding models, speech recognition, speech synthesis, image generation, and video generation all lean on the same set, with the details bent to fit each architecture. [Chapter 6](https://learn-inference.com/chapters/modalities) covers where the analogy holds and where it breaks.

---

# Scale changes the problem · Inference

<!-- https://learn-inference.com/chapters/inference/scale-changes-problem -->

Here is the part that catches teams out. The dominant inference problem is not fixed. It changes as you grow, and each new version of it is largely unrelated to the one you just finished solving.

A team that has spent six months becoming excellent at runtime tuning does not thereby become good at capacity planning. The skills barely overlap. What was a config file becomes a conversation with a vendor about what is physically available in a given region next quarter.

> Figure. What breaks next, by fleet size. Each stage has a different dominant problem, and solving one does not prepare you for the next. The work at the top of the ladder barely resembles the work at the bottom.Illustrative numbers

The last stage is the interesting one. Once workloads are spread across regions and providers, the failure mode stops being “we cannot get GPUs” and becomes “we have GPUs and cannot use them.” One cluster queues requests while another idles. The work at that point is unification: making everything you rent, everywhere, behave as a single pool of compute.

Spreading out has two side benefits worth naming. It protects you from any one region or provider having a bad day, and for a global product it puts inference physically closer to users, which shortens every round trip.

---

# The three layers · Inference

<!-- https://learn-inference.com/chapters/inference/three-layers -->

Doing this well means solving three separate problems, and they do not substitute for one another. A team can be excellent at one and still have a service nobody wants to depend on.

> Figure 0.1. The three layers of an inference stack. Each layer owns a different problem, and a team can be excellent at one while another quietly fails. The layers are ordered by distance from the hardware: the runtime sits closest to the GPU, the tooling closest to the engineer.

The **runtime** is the narrowest of the three and the easiest to reason about: one model, one instance, and the question of how much useful work you extract from the GPU you are already paying for. It rests on a deep software stack that runs from CUDA up through PyTorch to inference engines like vLLM, SGLang, and TensorRT-LLM. Low-level work matters here in a way that surprises people coming from web services; a better attention kernel is worth more than most application-level cleverness.

The **infrastructure** layer exists because no single instance is enough. Whatever you do to one server, it will eventually receive more traffic than it can serve, and that is not a CUDA or PyTorch problem. It is a distributed-systems problem: replication, routing, and scaling.

The **tooling** layer is the one teams skip, because it feels like overhead next to the other two. It is the interface everything else is operated through, and its absence shows up as a team that cannot deploy without a specific person being awake.

---

# Two phases, two disciplines · Inference

<!-- https://learn-inference.com/chapters/inference/two-phases -->

Every generative model passes through the same two phases. **Training** (Learning model weights from data. Happens once per model version, with a budget and a known end date.) is where weights are learned from data. **Inference** (Running a trained model to produce output for a user request. Happens continuously, at whatever rate your users demand.) is where those weights are put to work in production. They look adjacent on a diagram and they are almost nothing alike in practice.

Training is a project. It has a start, a budget, a cluster reserved for a known number of weeks, and a moment where it is finished. Inference is an operation. It has no end date, its load is set by other people, and its cost scales with your success. The failure modes differ too: a bad training run wastes its budget, while a bad inference service means the product itself is down.

Anyone who worked through the previous decade of machine learning already knows a version of this lifecycle. That version was easy. Serving a gradient-boosted tree meant a CPU, a Flask app, and a model file measured in megabytes. You could hold the whole thing in your head, and the interesting problems were all upstream in feature engineering.

Generative models broke that. The weights are large enough that where they sit in memory is an architectural decision. Output is produced one token at a time, so a single response is a few hundred sequential passes through the network rather than one. And demand is spiky in a way that batch scoring never was. You cannot take weights, rent a GPU, and expect the result to be fast or reliable enough to put in front of users. Something has to be built around it.

What this book means by inference

Not the mathematical sense of the word, and not the training-time forward pass. Inference here means the whole production concern: a model, the software serving it, the hardware underneath, and the systems that keep all of it available when traffic arrives.

---

# Modalities

<!-- https://learn-inference.com/chapters/modalities -->

Most non-text models are transformer variants, which means the toolkit transfers almost intact. Image and video generation are the exception, with a different architecture and different bottlenecks.

Vision-language models, multimodal embeddings, speech recognition, and speech synthesis all adapt LLM architectures. That is good news: the same engines serve them, the same quantization and batching and caching apply, and the intuitions from [Chapter 2](https://learn-inference.com/chapters/models) hold.

What changes is the metric. Time to first token is meaningless for a model whose output is an image, and the first 20 milliseconds of audio from a TTS model is not useful on its own. Each modality measures something different, and knowing which number matters is most of the work.

## 6 sections

1.  [6.1Vision language modelsVideo processing for vision language models · Omni-modal models](https://learn-inference.com/chapters/modalities/vlms)
2.  [6.2Embedding modelsEmbedding model architecture · Embedding model inference](https://learn-inference.com/chapters/modalities/embeddings)
3.  [6.3ASR modelsSingle-chunk latency optimization · Long file latency optimization · Diarization](https://learn-inference.com/chapters/modalities/asr)
4.  [6.4TTS modelsStreaming real-time text to speech · Speech-to-speech models](https://learn-inference.com/chapters/modalities/tts)
5.  [6.5Image generation modelsImage generation kernel optimization · One weird trick for faster image generation](https://learn-inference.com/chapters/modalities/image-models)
6.  [6.6Video generation modelsAttention optimization and quantization · Context parallelism](https://learn-inference.com/chapters/modalities/video-models)

---

# ASR models · Modalities

<!-- https://learn-inference.com/chapters/modalities/asr -->

Automatic speech recognition takes audio and produces text. Whisper is the dominant open model, and the largest variant is 1.55B parameters: small enough to transcribe faster than real time on a fraction of a modern GPU.

Whisper is encoder-decoder: the encoder processes a 30-second audio window, the decoder autoregressively emits text. That 30-second limit is a hard architectural constraint and it shapes both of the optimization problems below.

### 6.3.1 Single-chunk latency optimization

For live transcription, the number to watch is round-trip time for one chunk. Roughly 200 milliseconds is the target, because that is about human reaction time.

On an optimized engine there is little left to win at the runtime layer: the model is small and already fast. The gains are in orchestration, and the big one is streaming, implemented at the API server rather than the model.

A WebSocket carries audio in and text out continuously. A voice activity detection model watches the stream and segments it into chunks, and each chunk runs through ASR normally. Keeping consecutive chunks on the same GPU has a second benefit: the previous chunk’s output can prefix the next one, which improves transcription quality.

### 6.3.2 Long file latency optimization

An hour-long podcast is 120 thirty-second chunks, and the useful property is that they are independent. Transcription of a long file is an embarrassingly parallel problem, so the latency floor is set by how wide you are willing to fan out rather than by the length of the file.

This is the throughput-versus-latency trade from [Chapter 1](https://learn-inference.com/chapters/prerequisites/about-your-app) in its clearest form. The seams need care: a chunk boundary through the middle of a word produces a transcription error, so a VAD model cuts at silence rather than at fixed intervals, and the transcripts are stitched back together by timestamp.

### 6.3.3 Diarization

Diarization answers “who spoke when”, and it is a separate model from transcription. Running it means a second pipeline whose output has to be aligned with the first.

It is also harder to parallelize, because speaker identity is a property of the whole recording. A speaker in chunk 90 has to be recognized as the same person from chunk 3, which resists the chunk-independence that makes transcription easy to scale.

---

# Embedding models · Modalities

<!-- https://learn-inference.com/chapters/modalities/embeddings -->

Embedding models turn a variable-length input into a fixed-length vector. They power search, retrieval, clustering, and recommendation, and they are the quiet workhorse of production AI, running in high-volume, low-latency deployments behind search and retrieval.

### 6.2.1 Embedding model architecture

Either a small BERT-style encoder (under a billion parameters) or a repurposed LLM of a few billion, with the LLM-based family now the more capable of the two. Either way the output is pooled across positions into a single vector.

Small and encoder-only changes everything about serving. There is no autoregressive loop, so no decode phase and no KV cache. One forward pass produces the entire output.

### 6.2.2 Embedding model inference

Which makes embeddings almost purely compute-bound and almost perfectly batchable. There is no memory-bandwidth wall to work around, because you are not re-reading the weights per token.

The optimizations are correspondingly different: large batches, sequence-length bucketing so short inputs are not padded to the longest in the batch, and often a [fractional GPU](https://learn-inference.com/chapters/hardware/instances), since a 300M-parameter model cannot keep an H100 busy.

A high-performance pipeline usually parallelizes tokenization on the CPU alongside GPU inference. For a model this small, CPU-side preprocessing is a genuine bottleneck rather than a rounding error.

---

# Image generation models · Modalities

<!-- https://learn-inference.com/chapters/modalities/image-models -->

Here the analogy to LLMs finally breaks. Diffusion has no autoregressive loop, no KV cache, and no memory-bandwidth wall. It is compute-bound from the first step to the last, and its latency is step count times step cost, predictable before the request starts, which no LLM request ever is.

### 6.5.1 Image generation kernel optimization

With no cache to manage and no batching subtleties to exploit, optimization means making each denoiser pass cheaper. `torch.compile` and kernel fusion from [Chapter 4](https://learn-inference.com/chapters/software/cuda) do most of the work, and quantization helps because low-precision tensor cores are faster.

Note that quantization pays differently here. For LLM decode its main benefit is halved memory traffic; for diffusion, which is not bandwidth-bound, the win is the faster compute path.

### 6.5.2 One weird trick for faster image generation

Every diffusion step actually runs the denoiser twice: once guided by the prompt and once without, combined according to the guidance parameter. So a 50-step generation is 100 passes.

But the composition is decided early. After the first several steps the model has committed: it is not going to change its mind on step 40 and start drawing a dog when it has been drawing a cat. The remaining steps fill in detail, and detail does not need the prompt re-asserted.

> Figure 6.9. Turning guidance off partway through. Each step normally runs the denoiser twice, guided and unguided. Composition is settled early, so the guided pass stops paying for itself: drop it for the last 20 of 50 steps and you save a fifth of the work at the same step count.Constants from the book

Turning guidance off for the last stretch saves passes without reducing step count. Skip it for the final 20 steps of a 50-step run and you do 80 passes instead of 100, with quality generally holding. It is a hack, but it removes a fifth of the generation time without reducing step count.

---

# TTS models · Modalities

<!-- https://learn-inference.com/chapters/modalities/tts -->

Text to speech inverts ASR. Modern TTS is usually a fine-tuned LLM emitting audio tokens, which means most of the LLM toolkit applies: quantization, in-flight batching, the same optimized engines.

### 6.4.1 Streaming real-time text to speech

The metric is time to first audio chunk. Nobody needs the whole utterance before playback starts; they need enough to begin speaking, and the rest can arrive while the first clause plays.

Which means a well-built TTS stage can start producing audio from a partial LLM response. You do not wait for the sentence to finish; you start speaking the first clause while the model is still writing the second. That overlap is the difference between a voice agent that feels responsive and one that does not.

### 6.4.2 Speech-to-speech models

Speech-to-speech models collapse the three-stage pipeline into one, taking audio in and producing audio out without a text intermediate. Removing two stages removes two sets of latency and two failure modes, and preserves tone and emotion that transcription discards.

The costs are real: today’s speech-to-speech models are less capable and more expensive than the cascade, and you lose the text transcript most products need for logging and moderation. Worth it for some products, not for most.

Why voice is unforgiving

The budget covers three models plus the network, in series, inside a conversational turn. And voice is where users notice immediately, because everyone has a lifetime of calibration for how long a person takes to answer.

> Figure. The voice agent latency budget. Three models in series plus network, and the whole chain has to fit inside a conversational turn. No single stage is slow. The sum is the problem, which is why a voice pipeline is budgeted end to end rather than model by model. The marks are rules of thumb for when the pause becomes audible, not hard limits.Illustrative numbers

---

# Video generation models · Modalities

<!-- https://learn-inference.com/chapters/modalities/video-models -->

Video is diffusion with a time axis, at far higher cost. Attention now spans space and time together, so sequence lengths dwarf anything in text, and the quadratic term in attention becomes the dominant cost of the entire pipeline.

### 6.6.1 Attention optimization and quantization

Because attention dominates, the optimizations from [section 2.5](https://learn-inference.com/chapters/models/optimizing-attention) matter more here than anywhere else. Then comes a family of caching tricks with no LLM equivalent (reusing outputs across timesteps, or hidden states across layers), worth thirty to forty percent when quality holds.

Quantization is close to mandatory rather than optional. Dropping to lower-precision tensor cores doubles available FLOPS, and video generation needs every one of them.

### 6.6.2 Context parallelism

Video is the case where multi-GPU serves a single request rather than more users. Context parallelism splits the sequence itself across GPUs, each handling a slice of the space-time volume and exchanging what attention requires.

This inverts the usual economics. Elsewhere you add GPUs to serve more people; here you add GPUs so one request completes in a tolerable time, and the interconnect bandwidth from [Chapter 3](https://learn-inference.com/chapters/hardware/instances) is what decides whether that scales.

That covers the models. The last chapter is everything around them.

---

# Vision language models · Modalities

<!-- https://learn-inference.com/chapters/modalities/vlms -->

A **vision language model** (A model that accepts images alongside text and reasons over both in a single sequence.) bolts a vision encoder onto an LLM. Images are encoded into embeddings, projected into the language model’s embedding space, and spliced into the token sequence alongside the text.

From a serving perspective the important consequence is that images are expensive tokens. A single high-resolution image can occupy hundreds or thousands of sequence positions, which inflates prefill and inflates the KV cache. A VLM request with four images is not a slightly heavier text request; it is a different workload.

The encoder also runs before the LLM, adding a compute-bound stage ahead of a pipeline that was already compute-bound during prefill. Image preprocessing on the CPU can quietly become the bottleneck, which is a good reminder that the [whole instance](https://learn-inference.com/chapters/hardware/instances) matters and not just the GPU.

### 6.1.1 Video processing for vision language models

Video is sampled into frames, and each frame costs what an image costs. The entire engineering problem is deciding how few frames you can get away with.

Downsampling is practically obligatory, since most videos have long stretches where little changes: reduce the resolution and the frame rate until the entire clip fits in a single inference request. At these sequence lengths, the caching and attention optimizations from later chapters stop being optional.

### 6.1.2 Omni-modal models

Omni-modal models accept text, images, audio, and video and emit several of those too. Architecturally they are the same trick generalized: encoders per modality, everything projected into one sequence.

Serving them is harder than serving the sum of their parts, because request cost now varies enormously by input type. A batch of text requests and a batch of video requests are different workloads hitting the same endpoint, and scheduling has to account for it.

---

# Models

<!-- https://learn-inference.com/chapters/models -->

You cannot make something faster without knowing where its time goes. This chapter builds a model up from a matrix multiplication, then does the one calculation that tells you which resource you are actually short of.

The lineage is longer than the hype suggests. Perceptrons were built in hardware in the 1950s, abandoned, then revived as multi-layer networks once backpropagation gave them a way to learn. Depth arrived in the 2000s. AlexNet showed in 2012 that deep networks could do useful work and that GPUs were the way to run them.

The part that matters for us starts in 2017, when Vaswani and colleagues published “Attention Is All You Need.” A transformer is a neural network with an attention mechanism that learns relationships between positions in a sequence, and it turned out to generalize remarkably well. The models you will serve for text, images, audio, and video are nearly all transformer variants.

Two styles matter for inference, and they have opposite performance characteristics.

-   **Autoregressive generation** starts from a tokenized sequence and predicts the next token, repeatedly. LLMs work this way.
-   **Iterative denoising** starts from random noise and refines toward an output over a fixed number of steps. Image and video generation work this way.

## 5 sections

1.  [2.1Neural networksLinear layers and matmul · Activation functions](https://learn-inference.com/chapters/models/neural-networks)
2.  [2.2LLM inference mechanicsLLM architecture · Transformer blocks · Attention · Mixture of Experts models](https://learn-inference.com/chapters/models/llm-mechanics)
3.  [2.3Image generation inference mechanicsImage generation model architecture · Few-step image generation models · Video generation](https://learn-inference.com/chapters/models/image-mechanics)
4.  [2.4Calculating inference bottlenecksOps:byte ratio and arithmetic intensity · LLM inference bottlenecks · Image generation inference bottlenecks](https://learn-inference.com/chapters/models/bottlenecks)
5.  [2.5Optimizing attention](https://learn-inference.com/chapters/models/optimizing-attention)

---

# Calculating inference bottlenecks · Models

<!-- https://learn-inference.com/chapters/models/bottlenecks -->

A GPU offers two resources: compute, measured in operations per second, and memory bandwidth, measured in bytes per second. In an ideal world neither waits on the other. In reality one is saturated while the other idles, and which one it is determines what work is worth doing.

This matters more than any other diagnostic in inference. If an operation is bound by memory bandwidth, no amount of compute optimization will make it faster. You can double your FLOPS and measure no change at all.

Most inference systems break down like this:

-   **LLM prefill** is compute bound.
-   **LLM decode** is memory bound.
-   **Image and video generation** are compute bound.

### 2.4.1 Ops:byte ratio and arithmetic intensity

Divide a GPU’s compute by its bandwidth and you get its **ops:byte ratio** (A GPU's peak compute divided by its memory bandwidth. The arithmetic intensity at which that GPU is perfectly balanced.): how many operations it must perform per byte fetched to keep both resources busy. An H100 in FP16 does about 989 teraFLOPS against 3.35 TB/s, which is roughly 295 operations per byte.

The matching property of an algorithm is its **arithmetic intensity** (The ratio of arithmetic operations to bytes of memory traffic for a given computation.): operations performed divided by bytes moved. Compare the two numbers and you have your answer. Below the ratio, memory bound. Above it, compute bound.

> Figure 2.13. The roofline: where your bottleneck actually is. The diagonal is what memory bandwidth allows; the flat top is what the tensor cores allow. Their intersection is the ridge. Decode at batch size 1 sits far to the left of it on every GPU you can buy, which is why batching exists. Intensities here are first-order approximations for FP16 weights.Datasheet numbers

### 2.4.2 LLM inference bottlenecks

Now the two phases make sense. Prefill loads the model weights once and then does large matrix-matrix multiplications across the entire input sequence. Enormous arithmetic against a single read: high intensity, compute bound.

Decode loads the same weights again for every single token, and uses them for a comparatively tiny matrix-vector multiplication. A 70B model in FP16 means moving 140 GB from memory to produce one token. On an H100 that read alone takes about 42 milliseconds no matter how fast the arithmetic is.

Which is the entire argument for batching. If you are going to pay to read the weights, read them for thirty-two sequences instead of one. The bytes moved barely change, the useful work multiplies, and arithmetic intensity climbs toward the ridge. Batching does not make decode faster for any individual user: it makes the memory traffic you were already paying for do more work.

The one calculation worth memorizing

Divide your GPU’s TFLOPS by its TB/s to get the ridge. Estimate your operation’s FLOPs per byte. If you are below the ridge, stop looking at compute: every optimization that matters is about moving fewer bytes or getting more use out of each one.

### 2.4.3 Image generation inference bottlenecks

Diffusion sits on the other side of the ridge. Each denoising step is a large parallel operation over the full latent, with high arithmetic intensity throughout, so image generation is compute bound from start to finish.

That changes which levers work. Batching, which transforms LLM decode, does much less here: you are already saturating the tensor cores. The wins come instead from lower precision, better kernels, and doing fewer steps.

---

# Image generation inference mechanics · Models

<!-- https://learn-inference.com/chapters/models/image-mechanics -->

Image generation inverts the process. Rather than building output token by token, a diffusion model starts from pure noise and removes a little of it at each step, guided by your prompt, until an image is left behind.

The performance shape is completely different from an LLM. Step count is fixed before you start, so latency is predictable in a way autoregressive generation never is: you know at submission time roughly how long a request will take. And every step is a heavy parallel operation over the whole latent, so image generation is compute bound throughout rather than swinging between two regimes.

### 2.3.1 Image generation model architecture

A pipeline rather than a single network. A text encoder turns the prompt into embeddings; a denoiser (historically a U-Net, now increasingly a diffusion transformer) does the iterative work in a compressed latent space; a VAE decoder expands the final latent into pixels.

Working in latent space rather than pixel space is what makes this tractable at all: the denoiser operates on a representation dozens of times smaller than the image it eventually produces. Each component has its own performance profile, and each can be optimized, quantized, or scaled independently.

### 2.3.2 Few-step image generation models

If latency is step count times step cost, the most direct optimization is fewer steps. Distillation techniques have pushed models that once needed fifty steps down to four, or one.

This is the cleanest quality-for-speed dial in the whole book. It is also a product decision rather than an engineering one: a four-step model is not subtly worse in a way only benchmarks notice, it is visibly different, and whether that trade is acceptable depends on what the image is for.

### 2.3.3 Video generation

Video is diffusion with a time axis, and the cost is brutal. Attention now spans both space and time, so a few seconds of video involves sequence lengths that dwarf anything in text.

This is where inference stops being a single-GPU concern by default. A single video request can require several GPUs cooperating, which is unusual: for most workloads multi-GPU is about serving more users, not about serving one at all.

---

# LLM inference mechanics · Models

<!-- https://learn-inference.com/chapters/models/llm-mechanics -->

Text does not enter a model as text. A tokenizer splits it into subword units (common words get one token, rarer ones get broken into pieces), and each becomes an integer index into a vocabulary. One token is roughly three-quarters of an English word.

Every request assembles into a single sequence: the input, an optional reasoning sequence for models that think before answering, and the output. Together these have to fit inside the model’s **context window** (The maximum number of tokens a model can process and generate for a single request, input and output combined.).

Getting from a chat request to that single sequence is the job of the chat template, which encodes roles, tool signatures, and multimodal inputs into the exact format the model saw during training. Templates differ subtly between models, and a template applied slightly wrong degrades output rather than visibly breaking, which is exactly why the engine has to implement it correctly.

Once tokenized, inference runs in two phases.

-   **Prefill** processes the entire input sequence at once, computing attention for every input token and storing the results in the **KV cache** (Cached key and value tensors from the attention computation, kept so that previously-processed tokens do not have to be recomputed on every subsequent forward pass.). This determines time to first token.
-   **Decode** then runs one forward pass per output token, autoregressively, each pass attending over everything generated so far. This determines tokens per second.

Each decode pass has to end in a token, and neural networks emit vectors. The output layer produces one **logit** (An unnormalized score, one per vocabulary entry, produced by the model's output layer before softmax.) per vocabulary entry (tens or hundreds of thousands of them), which normalize into a probability distribution over every token the model knows.

> Figure 2.6. Turning logits into one token. The model emits one score per vocabulary entry; sampling decides which one becomes text. Temperature rescales the whole distribution, top-k truncates by rank, and top-p truncates by cumulative mass, which is why top-p adapts to the model's confidence and top-k cannot.Illustrative numbers

Setting temperature to 0, or top-k to 1, makes selection deterministic: the highest-probability token wins every time. Structured output pushes this further with logit biasing, masking tokens that would break a JSON schema before sampling ever happens. That machinery is what makes reliable tool calling possible, and implementing it correctly is squarely an inference concern.

Generation continues until the model emits a stop token, or until the context window or a `max_tokens` limit intervenes.

### 2.2.1 LLM architecture

Every model on Hugging Face ships a `config.json`: a few dozen lines fixing layer count, hidden dimensions, head counts, vocabulary size. Architecture is the set of decisions frozen at training time, and it determines what inference optimizations are even available to you.

Within one architecture there can be:

-   **Multiple sizes**, like an 8B and a 70B of the same family.
-   **Multiple variants**, like base and instruct, which share an architecture entirely.
-   **Unlimited fine-tunes.** LoRA and friends change behaviour without changing architecture, which is precisely why you can serve many of them from one base model.

### 2.2.2 Transformer blocks

A transformer is the same block repeated dozens of times. Each block holds an attention mechanism, a feed-forward network, normalization layers, and residual connections that let information skip past the block entirely.

The repetition is convenient for us. Optimize one block and you have optimized all eighty, which is why kernel-level work pays back so disproportionately here compared to typical application code.

### 2.2.3 Attention

Attention lets every token look at every other token and decide what is relevant. Three matrices (queries, keys, and values) are derived from the input, and the mechanism scores each query against every key, normalizes the scores, and uses them to take a weighted sum of values.

The important structural fact for inference is that scoring every position against every other is quadratic in sequence length. Double the context and attention’s intermediate matrices quadruple. That single property drives most of [section 2.5](https://learn-inference.com/chapters/models/optimizing-attention), a good part of [Chapter 5](https://learn-inference.com/chapters/techniques), and the existence of FlashAttention.

It is also why the KV cache exists. Without it, every decode step would recompute keys and values for the entire sequence so far, turning a linear generation into a quadratic one.

> Figure. Where the VRAM goes. Weights are fixed the moment you pick a model and a precision. The KV cache is not: it grows with context length and with every concurrent request, and it is what actually caps your batch size.Illustrative numbers

### 2.2.4 Mixture of Experts models

A Mixture of Experts model replaces the feed-forward network in each block with many parallel experts and a router that picks a few per token. A 235-billion-parameter MoE might activate just twenty-two billion parameters for any given token.

This is a genuinely strange object from a serving perspective. Total parameters set your memory requirement, because every expert must be resident somewhere. Active parameters set your compute and bandwidth per token. So for a single request an MoE is cheap to run and expensive to hold. Under batching, though, different requests light up different experts, and most of the model ends up active anyway.

It also creates a routing problem that does not exist for dense models: which GPU holds the expert this token needs, and what does it cost to get there. That question is the whole subject of expert parallelism in [Chapter 5](https://learn-inference.com/chapters/techniques/parallelism).

---

# Neural networks · Models

<!-- https://learn-inference.com/chapters/models/neural-networks -->

Strip away the vocabulary and a neural network is a stack of layers that each take a vector, multiply it by a matrix of learned weights, and pass the result on. An input layer, many hidden layers, an output layer.

### 2.1.1 Linear layers and matmul

A linear layer computes `y = Wx + b`: a matrix of weights times an input vector, plus a bias. That single operation is where essentially all of the arithmetic in inference happens, which is why GPUs (machines built to multiply matrices) turned out to be the right hardware for a problem nobody designed them for.

It also explains why parameter count is such a good proxy for cost. More parameters means bigger matrices, which means more multiply-accumulates and, more importantly, more bytes to move.

### 2.1.2 Activation functions

Stack two linear layers and you have accomplished nothing. The composition of two matrix multiplications is another matrix multiplication, so a hundred stacked linear layers collapse algebraically into one. Depth would buy no expressiveness at all.

An activation function (ReLU and its relatives) applies a non-linearity between layers, which is what stops the collapse. It is a small operation with an outsized structural role: without it there is no such thing as a deep network.

---

# Optimizing attention · Models

<!-- https://learn-inference.com/chapters/models/optimizing-attention -->

Attention is the most expensive operation in a transformer and the most studied. The naive implementation is three lines: score queries against keys, normalize, weight the values. Each line reads from memory, computes, and writes back.

The problem is the intermediate matrices. For a 4,096-token sequence the score matrix is 4,096 by 4,096 (about 32 MB in FP16, roughly a high-resolution RAW photograph), and the naive version writes it to memory and reads it back, twice, for every head in every layer. The arithmetic is not the expensive part. The round trips are.

FlashAttention’s insight was that you can avoid materializing that matrix at all. Tile the computation, keep tiles in on-chip SRAM, and compute the softmax incrementally so the full score matrix never touches HBM. Same mathematical result, a fraction of the memory traffic. It is the canonical example of the principle that reorganizing memory access beats optimizing arithmetic.

Beyond exact attention, several approaches trade fidelity for time:

-   **Grouped-query attention** shares key and value heads across multiple query heads, shrinking the KV cache several times over. You saw its effect in the calculator in [section 2.2](https://learn-inference.com/chapters/models/llm-mechanics).
-   **Sliding window attention** limits each token to a fixed window of neighbours, making cost linear in sequence length rather than quadratic.
-   **Sparse and gated attention** approximate the full computation by attending selectively, trading exactness for scaling.

These are architectural decisions made at training time, not knobs you turn at inference. They belong here because they explain why two models with identical parameter counts can have wildly different serving costs, and why [model selection](https://learn-inference.com/chapters/prerequisites/model-selection) is the highest-leverage performance decision you make.

The bottleneck is as much a fact about the hardware as about the model, and what that hardware actually offers is the subject of the next chapter.

---

# Prerequisites

<!-- https://learn-inference.com/chapters/prerequisites -->

Optimization means picking the best option from a set. Until your product defines what “best” means, performance work has no criterion to optimize against, and a benchmark cannot supply one.

Almost nothing in inference is a pure win. You buy throughput with latency, latency with money, and money with quality. Which of those trades is acceptable depends entirely on what you are building, and that makes the product questions genuine prerequisites rather than preamble.

The useful frame is specialization. Football players are big, fast, and strong, but they are not as big as sumo wrestlers, as fast as sprinters, or as strong as powerlifters. They are built for the specific demands of a position across a season. An inference service is the same: the more constraints you can name, the better it can be made. A system that must serve every workload well serves none of them excellently.

Before optimizing anything, you should be able to answer five questions.

-   **Which model?** Everything downstream depends on this one, and it is the decision with the largest effect on both cost and speed.
-   **What is the interface?** How inputs arrive and what shape the output has to take.
-   **What is the latency budget?** End to end, from the user’s action to something appearing on their screen.
-   **What are the unit economics?** What a request, a user, or a month can reasonably cost.
-   **What do usage patterns look like?** Concurrency, and whether it follows business hours, a launch, or nothing at all.

Early on, most of these have no honest answer yet. That is a fine reason to use an off-the-shelf API and leave inference engineering alone until the questions have answers. The work is worth doing once the constraints are real.

## 4 sections

1.  [1.1Scale and specialization](https://learn-inference.com/chapters/prerequisites/scale-and-specialization)
2.  [1.2About your appAI-native applications · Online versus offline · Consumer versus B2B](https://learn-inference.com/chapters/prerequisites/about-your-app)
3.  [1.3Model selectionModel evaluation · Fine-tuning for domain-specific quality · Distillation](https://learn-inference.com/chapters/prerequisites/model-selection)
4.  [1.4Measuring latency and throughputLatency percentiles · End-to-end metrics](https://learn-inference.com/chapters/prerequisites/latency-throughput)

---

# About your app · Prerequisites

<!-- https://learn-inference.com/chapters/prerequisites/about-your-app -->

Every decision that follows is downstream of your use case. A basketball coach and a gymnastics coach walk into the same school and recruit opposite children; neither is wrong.

Two situations require building for full generality: you have trained a foundation model and are selling access to it, or you are building an inference platform that has to serve models you have not seen yet. Everything else is a vertical application, and vertical applications should be greedy about constraints.

### 1.2.1 AI-native applications

The categories below are a small sample, but they show how far apart the requirements sit. Each one is trying to make a different number small.

> Figure. What each kind of product optimizes for. Describe what you are building with the two controls; the figure answers with the number to make small. The archetypes light up as they match. Search lights in more than one place because it is two workloads sharing one name, and three products share the person-and-one-call shape, separated only by what failure means.

### 1.2.2 Online versus offline

The central trade in inference is **latency** (How long one request takes, measured from the user's perspective.) against **throughput** (How much total work the service completes per unit time, across all users.). Lower latency makes the product feel faster. Higher throughput makes it cheaper, because the same traffic fits on fewer GPUs.

Most applications are online. Somebody is waiting, so latency wins. But a surprising amount of real inference work has nobody waiting at all: transcribing a back catalogue of podcasts, re-embedding a document corpus on a schedule, cleaning data for a training run. These are batch jobs, and for batch jobs an individual request being slow is irrelevant as long as the system chews through the queue.

The same model often serves both. Whisper might power a live dictation feature and a bulk transcription pipeline. If both have real volume, run two deployments of the same weights with opposite configurations rather than one compromise that is mediocre at each.

### 1.2.3 Consumer versus B2B

Consumer products are cost-sensitive and their traffic is unpredictable. Many are built for virality, which means a single post can multiply your load overnight. Optimize for marginal cost and for the ability to scale sideways in a hurry; keep latency and availability respectable rather than exceptional.

B2B products usually have better margins and steadier load, and in exchange they are held to a much higher standard. Software sitting in someone else’s revenue path does not get to be slow or occasionally unavailable. Favour latency and uptime; treat cost as the important secondary concern it is.

Compliance narrows the options

In regulated industries this stops being an afterthought and starts constraining which regions you may put GPUs in, what may be logged, and which providers you may use at all. Data sovereignty, user privacy, and your providers’ own compliance posture all limit the search space before performance work begins. Get security and legal involved early rather than rebuilding later.

---

# Measuring latency and throughput · Prerequisites

<!-- https://learn-inference.com/chapters/prerequisites/latency-throughput -->

Two numbers dominate LLM serving. **TTFT** (Time to first token: how long the user waits before any output appears.) is how long before the first token appears, and it is governed by prefill, which is compute bound. **TPS** (Tokens per second: the rate at which output arrives after the first token.) is how fast tokens arrive afterwards, governed by decode, which is memory bandwidth bound. Different phases, different bottlenecks, different fixes.

> Figure 1.4. Time to first token versus tokens per second. Two numbers, two phases, two bottlenecks. A response that starts instantly and trickles can feel faster than one that pauses and then dumps, even when the second finishes first.Illustrative numbers

TTFT is unambiguous. TPS is not, and the ambiguity causes real confusion in benchmark comparisons, because the same three letters describe both a latency metric and a throughput metric. When precision matters, say which you mean:

-   **Perceived TPS** is tokens per second for one user. A latency metric.
-   **Total TPS** is tokens per second across the whole service. A throughput metric, and typically far larger.
-   **Inter-token latency** is the gap between consecutive tokens. 10 ms of ITL is 100 tokens per second per user.

Both metrics assume streaming output and a human reading it. When the output is a tool call an agent will parse, individual tokens have no value on their own and the only number that means anything is total response time.

### 1.4.1 Latency percentiles

Mean latency is the most commonly reported figure and a poor summary of the distribution. Response times are right-skewed: most cluster near a mode, and a long tail stretches out to the right. The mean gets dragged toward the tail without describing it, so it simultaneously overstates the typical experience and understates the bad one.

> Figure 1.5. Why the mean hides your worst requests. Response times cluster and then trail off to the right. Push the tail weight up and watch the mean barely move while P99 runs away. That gap is the one in a hundred users who thinks your product is broken.Constants from the book

Percentiles say something checkable instead. P90 means one request in ten is slower than this; P99 means one in a hundred. Those outliers do disproportionate damage, because a product where nine interactions feel instant and the tenth hangs for several seconds gets remembered as unreliable rather than fast.

Pushing the mean down is worth doing. Pushing P90 and P99 down is what makes a product feel dependable, and it is usually the harder and more valuable work.

### 1.4.2 End-to-end metrics

The last distinction is where you start and stop the clock. Inference time is on-GPU time: how long generation itself took. End-to-end time is what the user experienced, including network round trips, queueing, load balancer hops, and whatever your client code does before and after.

Keep both. Inference time tells you whether your model performance work is landing. End-to-end time tells you whether any of it reached the user.

The gap between them is diagnostic. When inference time is excellent and end-to-end time is poor, more kernel work will not help you: the problem is in the infrastructure, and [Chapter 7](https://learn-inference.com/chapters/production) is where to look.

---

# Model selection · Prerequisites

<!-- https://learn-inference.com/chapters/prerequisites/model-selection -->

Hold hardware, runtime, and optimizations constant, and a smaller model is faster and cheaper than a larger one. Nothing else you do moves the numbers as much.

Which makes model choice, not engine choice or speculation algorithm, the highest-leverage decision in inference performance. It is also the one most often treated as settled before the performance conversation starts.

Before product-market fit, use a frontier model through an API and do not think about this. After it, invert the advice completely: find or build the smallest model that still passes your evaluations. Sometimes that is still a trillion-parameter frontier model. It is always worth checking whether it is not.

Architecture choice has a second, less obvious effect. Inference engines support architectures unevenly, and an unusual model can quietly cut you off from half the optimization toolkit. Popular architectures come with better support across every engine, which is worth real performance.

### 1.3.1 Model evaluation

Evaluations measure whether a model is good at your task. They are a prerequisite for inference work for two reasons: they stop you spending a month making a useless model fast, and they give you the baseline you need before applying any technique that might cost quality.

That second reason is the one people skip. Quantization is the big one (the lossy member of [Chapter 5](https://learn-inference.com/chapters/techniques)'s toolkit), and without a baseline you will not notice the degradation until a user does.

Public benchmarks like MMLU or SWE-bench are useful for drawing up a shortlist and useless for anything after that. They saturate, and labs have every incentive to optimize against them directly. Goodhart’s law applies with unusual force here: when a measure becomes a target, it stops being a good measure.

Three things make evaluation work actually pay off. Look at your own data rather than only at aggregate scores. Be precise about the hardest cases your model must handle and concentrate there. And use existing tooling, because this is a well-trodden problem and rebuilding it is not where your advantage lies.

### 1.3.2 Fine-tuning for domain-specific quality

Fine-tuning adapts a pre-trained model to a narrower task by training it further on new data. The architecture stays the same; the weights move.

For inference this is a lever on size. If a small fine-tuned model passes evaluations that only a large general model passed before, every latency and cost target afterwards becomes easier.

Text-to-SQL is the standard example, and it is close to the best case. General coding models write good SQL, but they are hundreds of billions of parameters and know a hundred other languages you are not using. SQL is small and highly structured, so a few-billion parameter model fine-tuned on it can match them on that one task.

Most domains will not compress that far. What the example shows is what becomes possible when the domain is genuinely narrow, the evaluation criteria are sharp, and you have clean labelled data. Miss any of those three and fine-tuning tends to disappoint.

### 1.3.3 Distillation

Distillation trains a small student model to imitate a large teacher. It differs from fine-tuning on synthetic data in what the student sees: not just the teacher’s final answers, but its full probability distributions over tokens. That is a much richer signal, and it transfers behaviour rather than facts.

The distinction is worth holding onto. Fine-tuning makes a model better in a domain. Distillation makes a model behave like another model, including the parts of that behaviour you would rather it did not copy.

In practice distillation is far less common than fine-tuning. When a lab ships a family of sizes, the small members are usually trained independently rather than distilled, specifically so the large model’s biases do not put a ceiling on them. Where it earns its keep is when a lab has trained only a frontier-scale model and wants it to reach hardware it otherwise never would.

DeepSeek-R1 is the well-known case. At 671 billion parameters the model itself was out of reach for most people, so alongside it DeepSeek released versions distilled onto Llama 3 and Qwen 2.5 architectures. Those distills scored lower on benchmarks but showed similar reasoning behaviour, and because they sat on popular architectures they inherited all the existing performance work for free. That last part is the underrated half of the decision.

---

# Scale and specialization · Prerequisites

<!-- https://learn-inference.com/chapters/prerequisites/scale-and-specialization -->

There are two ways to get a model into your product.

-   **Shared inference:** send traffic to a public endpoint and pay per million tokens.
-   **Dedicated deployment:** rent GPUs, run the service yourself, and pay per hour whether or not anyone is using it.

This is a related but separate axis from open versus closed weights. Plenty of open models are available on shared endpoints, and large customers can often get dedicated capacity for closed models. The connection is that open weights are what make unrestricted dedicated inference possible at all.

Almost every product should start shared. Zero overhead, no cold starts, no capacity planning, and you only pay for what you use. What you give up is control: your uptime is capped by theirs, your latency is whatever they decide, and your cost scales linearly forever.

That last point is the one that eventually forces the move. Linear cost is a bargain at low volume and a problem at high volume, and the crossover comes later than a raw per-token comparison suggests, because a dedicated GPU bills for the hours you are asleep too.

> Figure. When per-token billing stops being the cheap option. Shared endpoints cost nothing at zero and scale linearly. Dedicated GPUs are a step function: flat until you saturate a card, then another card. Drop utilization to what your traffic really looks like overnight and the crossover moves a long way right.Illustrative numbers

Utilization is the term to keep in view. A GPU that is busy 20 percent of the day costs the same as one busy 100 percent of the day, so a napkin calculation using peak throughput will tell you to migrate before you should. Three things actually justify the switch:

-   **Scale:** your volume makes per-GPU cheaper than per-token, honestly accounting for idle time.
-   **Specialization:** you are running a fine-tuned model, or you have latency or uptime requirements a shared endpoint will not meet.
-   **Orchestration:** your product chains several models together and the network hops between them have become the latency.

Moving makes you responsible for your own inference engineering. That is more control and more surface area, and it raises the floor on your monthly spend from zero to the cost of the smallest sensible fleet. Do it when there is a clear business reason, not because it feels more serious.

---

# Production

<!-- https://learn-inference.com/chapters/production -->

A perfectly optimized runtime that cannot get a GPU when traffic arrives is a slow service. This chapter is everything around the model, which is where most production incidents actually live.

The techniques in the previous chapters make one replica fast. Nothing in them helps when you need forty replicas in ninety seconds, or when your region is out of H100s, or when a deploy takes the service down for eight minutes. Those are different problems with different answers.

## 6 sections

1.  [7.1ContainerizationDependency management · NIMs](https://learn-inference.com/chapters/production/containerization)
2.  [7.2AutoscalingConcurrency and batch sizing · Cold starts · Routing, load balancing, and queueing · Scale to zero · Independent component scaling](https://learn-inference.com/chapters/production/autoscaling)
3.  [7.3Multi-cloud capacity managementGPU procurement · Geo-aware load balancing · Building for reliability · Security and compliance](https://learn-inference.com/chapters/production/multi-cloud)
4.  [7.4Testing and deploymentZero-downtime deployment · Cost estimation · Observability](https://learn-inference.com/chapters/production/testing-deployment)
5.  [7.5Client codeClient latency overhead · Asynchronous inference · Streaming and protocol support](https://learn-inference.com/chapters/production/client-code)
6.  [7.6Where this leaves you](https://learn-inference.com/chapters/production/closing)

---

# Autoscaling · Production

<!-- https://learn-inference.com/chapters/production/autoscaling -->

Autoscaling adjusts how many replicas a model has. There are two signals to drive it from, and they disagree more often than you would expect.

-   **Utilization**: GPU memory and compute usage. It is a lagging indicator: by the time utilization is high, the queue has already formed.
-   **Traffic**: requests in the system. Available earlier, so it can be acted on proactively.

They come apart under uneven load. A handful of requests with hundreds of thousands of uncached input tokens will pin utilization while traffic counts look calm; a flood of small requests with high cache hit rates will do the opposite. Use both.

Five knobs shape a traffic-based autoscaler:

-   **Min replicas**, the floor that stays running regardless.
-   **Max replicas**, the ceiling when traffic is high.
-   **Autoscaling window**, the sliding period over which traffic is measured.
-   **Scale-down delay**, how long to wait before releasing capacity in case the spike returns.
-   **Concurrency target**, how many requests one replica should handle at once.

> Figure 7.4. Autoscaling across a spiky day. The gap between the two lines is queued requests. Cold start time decides how wide it gets: push it up and no scaling policy can keep the fleet ahead of a spike, which is why cold starts get their own section.Illustrative numbers

Every one of those settings is a bet. A longer scale-down delay protects you from flapping on spiky traffic and costs money when traffic genuinely cools. There is no correct value, only a value matched to your traffic shape and your tolerance for each failure.

### 7.2.1 Concurrency and batch sizing

The concurrency target is where autoscaling meets [batching](https://learn-inference.com/chapters/models/bottlenecks), and it is the setting people most often get backwards.

Set it too low and every replica runs at batch sizes that waste the GPU: you are back at the left side of the roofline, paying for tensor cores that idle while weights stream from memory. Set it too high and requests queue inside a replica, which is invisible to an autoscaler watching replica count and plainly visible to users watching p99.

Find it empirically. Sweep concurrency against latency for your actual workload and pick the point just before latency starts climbing steeply. That knee is your target.

### 7.2.2 Cold starts

A cold start is the time to bring up a new replica, and it quietly governs the whole autoscaling system. If you cannot add capacity quickly, you cannot scale down confidently, so you over-provision permanently and pay for it every hour of every day.

> Figure 7.9. What a cold start is actually made of. Four stages, each attacked separately (the durations here are representative, not measured). The mitigations are not equally valuable: a warm pool removes the one stage you do not otherwise control, because procurement speed is mostly a function of your cloud provider.Illustrative numbers

Attack the four stages separately, because they respond to completely different fixes. Procurement is solved with a warm pool of nodes you flex between models. Image loading is solved by making the image smaller and by giving the node more bandwidth to pull it. Weight loading is solved by streaming from a local cache rather than pulling from object storage. Engine startup is solved by caching the compiled artifact, which is the difference between seconds and minutes on TensorRT-LLM.

### 7.2.3 Routing, load balancing, and queueing

Round-robin is the wrong default for inference. Requests are wildly uneven (a 200-token request and a 200,000-token request are not interchangeable), so balancing on connection count sends work to replicas that are already deeply busy.

Better signals are queue depth and, where it applies, [cache locality](https://learn-inference.com/chapters/techniques/caching). Routing a request to the replica that already holds its prefix can matter more than routing it to the least busy one, which is a unusual property of this workload.

Queue somewhere. Without a queue, a spike becomes dropped connections; with one, it becomes latency you can observe and reason about. Bound it, and shed load deliberately when the bound is hit: a fast rejection is a better product experience than a request that hangs for ninety seconds and then fails.

### 7.2.4 Scale to zero

Scaling to zero means paying nothing when nobody is using a model, and paying a full cold start for whoever arrives first.

Correct for development environments, internal tools, and the long tail of models that see a handful of requests a day. Wrong for anything user-facing where that first request is somebody’s first impression. The economics only work when your cold start is fast.

### 7.2.5 Independent component scaling

Multi-model pipelines (the voice stack from [Chapter 6](https://learn-inference.com/chapters/modalities/tts), or a retrieval system with an embedder and a reranker and an LLM) should not scale as one unit.

Each component has its own throughput and its own hardware fit. Whisper is happy on a fractional GPU; the LLM needs a whole one. Bundling them means scaling all three because one is busy, which wastes the other two. Deploy separately and let each follow its own demand. This is [disaggregation](https://learn-inference.com/chapters/techniques/disaggregation) applied to a pipeline rather than to phases of one model.

---

# Client code · Production

<!-- https://learn-inference.com/chapters/production/client-code -->

The last few hundred milliseconds are on the client, and they are routinely ignored by teams who have spent months on the server.

### 7.5.1 Client latency overhead

TLS handshakes, DNS lookups, connection setup, and serialization all happen before your model sees anything. Re-use connections, keep them alive, and put a CDN or edge terminator close to users so the handshake happens near them rather than near your GPUs.

None of this is interesting work. It is regularly worth more milliseconds than the kernel optimization it gets skipped in favour of.

### 7.5.2 Asynchronous inference

Not every request needs an open connection. Batch jobs, long video generation, and bulk transcription are better served by submitting work, getting an identifier, and collecting the result later via polling or a webhook.

This decouples client timeouts from server duration entirely, which removes a whole category of failure. It also lets the scheduler defer that work to fill capacity that would otherwise idle.

### 7.5.3 Streaming and protocol support

Server-sent events are the default for token streaming: unidirectional, simple, and well-supported everywhere. WebSockets add bidirectionality, which is what live transcription and voice agents need: audio flowing up while text or audio flows down. gRPC gives bidirectional streaming with structured schemas, at the cost of more client-side machinery.

Streaming is the highest-leverage perceived-latency change available, and it happens entirely at the API layer. Nothing about the model changes; the user starts seeing output at your TTFT instead of your total response time. For a long response that is the difference between a product that feels fast and one that feels broken, which is where this book started, in [Chapter 1](https://learn-inference.com/chapters/prerequisites/latency-throughput).

---

# Where this leaves you · Production

<!-- https://learn-inference.com/chapters/production/closing -->

Three layers, and you now have all of them. The [runtime](https://learn-inference.com/chapters/models), where arithmetic intensity tells you which resource you are short of and the techniques in [Chapter 5](https://learn-inference.com/chapters/techniques) spend one to buy another. The infrastructure, where the problem changes shape every time you grow an order of magnitude. And the tooling, which decides whether any of it is operable by more than one person.

The specifics will age. Hardware generations, engine version numbers, and benchmark figures all have a shelf life measured in months. The constraints underneath move far more slowly: memory bandwidth has bounded decode for years, attention has been quadratic since 2017, and no amount of software makes a byte arrive faster than the bus allows.

The durable skill is finding the bottleneck. Everything else in this book can be re-derived from it.

---

# Containerization · Production

<!-- https://learn-inference.com/chapters/production/containerization -->

Inference ships as a container image: the engine, the CUDA libraries, the Python dependencies, and your serving code. That image is the unit of deployment, and its size directly sets one of the four terms in your cold start.

GPU images are enormous. CUDA and PyTorch alone run to several gigabytes before you add anything, and a naive image runs to many more. Most of those bytes are not strictly necessary.

### 7.1.1 Dependency management

This stack is unusually brittle about versions. CUDA, the driver, PyTorch, and the engine all have to agree, and the failure mode is frequently a runtime error deep inside a kernel launch rather than a clean import failure.

Pin everything, exactly, including transitive dependencies. The cost of a fully reproducible image is a slower upgrade path. The cost of an unpinned one is a deploy that works on Tuesday and does not on Wednesday for reasons nobody changed.

### 7.1.2 NIMs

NVIDIA Inference Microservices are prebuilt containers: a model, an optimized engine, and an API, packaged together and tuned for specific GPUs.

They are the abstraction trade from [Chapter 0](https://learn-inference.com/chapters/inference/abstraction) in container form. You get a working optimized deployment without doing the optimization, and you give up the ability to change anything the packager did not expose. Good for standard models on standard hardware; frustrating the moment you need something specific.

---

# Multi-cloud capacity management · Production

<!-- https://learn-inference.com/chapters/production/multi-cloud -->

Past a few hundred GPUs the problem stops being “how do I scale” and becomes “where are there GPUs”. That is a different discipline, and it is where inference stops being a software problem.

### 7.3.1 GPU procurement

Capacity is contended and geographically lumpy. On-demand pricing is highest and most flexible; reserved capacity is cheaper and commits you; spot is cheapest and can be reclaimed on minutes of notice.

Most serious deployments blend all three: reserved for baseline, on-demand for normal variation, spot for batch work that tolerates interruption. And they hold relationships with several providers, because the answer to “can I have forty more H100s today” varies by vendor and by week.

### 7.3.2 Geo-aware load balancing

Network latency is physics. A round trip from Sydney to us-east-1 costs more than 200 milliseconds before your model does anything, which for a [voice agent](https://learn-inference.com/chapters/modalities/tts) has already spent the entire budget.

Route users to their nearest region with capacity, and accept that those two conditions conflict. Falling back to a distant region beats queueing locally: a slower answer is better than no answer, and this is exactly the tradeoff a geo-aware balancer exists to make.

### 7.3.3 Building for reliability

Closed-model APIs sit around two nines. Dedicated deployments can reach four, and the difference is almost entirely architectural rather than about any single component being better.

Spread across regions and providers so no single failure takes everything. Health-check on real inference rather than on a process being alive: a replica in a bad CUDA state will answer your `/health` endpoint cheerfully while failing every actual request. Fail over automatically, and rehearse it.

### 7.3.4 Security and compliance

Where GPUs physically sit becomes a legal question in regulated industries. Data sovereignty may forbid a region outright, regardless of capacity or latency, and that constraint is not negotiable by engineering.

Beyond placement: encrypt in transit and at rest, be deliberate about what inference logs retain (prompts routinely contain exactly the data you promised not to store), and confirm your providers hold the certifications your customers will ask about. Involve security and legal before you design around a region, not after.

---

# Testing and deployment · Production

<!-- https://learn-inference.com/chapters/production/testing-deployment -->

### 7.4.1 Zero-downtime deployment

Deploy strategies written for stateless web services assume replicas start in seconds. Inference replicas take minutes, and any strategy that ignores that trades capacity for downtime.

Bring the new version up, send it a small slice of live traffic, verify it serves real requests, and ramp while the old fleet scales down underneath it. Running two full fleets side by side buys the same safety at twice the GPU bill, which is why canaries won this argument. And watch outputs, not just error rates: quality regressions from a new model or a new quantization show up in answers.

### 7.4.2 Cost estimation

Cost per token is the number the business cares about, and it falls out of three things you already know: GPU hourly rate, achieved throughput, and utilization.

Utilization is the term people omit, and it is the one that dominates. A GPU busy 30% of the day costs three times per token what the same GPU busy 90% of the day does, with identical hardware and identical optimization. Most inference cost problems are utilization problems rather than performance problems, which is the same point the [crossover calculator](https://learn-inference.com/chapters/prerequisites/scale-and-specialization) in Chapter 1 makes.

### 7.4.3 Observability

Instrument the numbers from [Chapter 1](https://learn-inference.com/chapters/prerequisites/latency-throughput) at percentiles, not averages: TTFT, inter-token latency, total response time, queue depth, batch size, cache hit rate, GPU utilization, and cost per token.

Two things separate useful inference observability from generic service monitoring. Measure end-to-end and on-GPU separately: the gap between them tells you whether to look at the model or the infrastructure. And monitor output quality, not just latency: a model that degrades after a config change produces perfectly fast, perfectly available, subtly worse answers, and nothing in a standard dashboard will tell you.

The alert that matters

Queue depth. It rises before latency does, which makes it the earliest honest signal that capacity is short. By the time p99 moves, users have already had the bad experience.

---

# Further reading

<!-- https://learn-inference.com/chapters/reading -->

The primary sources. This field moves fast enough that the papers are often more current than anything written about them, and most of these are readable in an evening.

## Architecture

The papers the field is built on, from the original transformer through diffusion and multimodal encoders.

-   [“Attention is All You Need”by Ashish Vaswani et al. (Neural Information Processing Systems, 2017)](https://arxiv.org/abs/1706.03762)
-   [“BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”by Jacob Devlin et al. (North American Chapter of the Association for Computational Linguistics, 2019)](https://arxiv.org/abs/1810.04805)
-   [“BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models”by Junnan Li et al. (International Conference on Machine Learning, 2023)](https://arxiv.org/abs/2301.12597)
-   [*Deep Learning*by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (The MIT Press, 2016)](https://www.deeplearningbook.org/)
-   [*Deep Learning with Python (2nd Edition)*by François Chollet (Manning, 2021)](https://www.manning.com/books/deep-learning-with-python-second-edition)
-   [“Denoising Diffusion Probabilistic Models”by Jonathan Ho et al. (ArXiv abs/2006.11239, 2020)](https://arxiv.org/abs/2006.11239)
-   [“DiT: Scalable Diffusion Models with Transformers”by William Peebles and Saining Xie (International Conference on Computer Vision (ICCV), 2023)](https://arxiv.org/abs/2212.09748)
-   [“FlashAttention: Fast and Memory-Efficient Exact Attention with IO Awareness”by Tri Dao et al. (ArXiv abs/2205.14135, 2022)](https://arxiv.org/abs/2205.14135)
-   [“FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning”by Tri Dao (ArXiv abs/2307.08691, 2023)](https://arxiv.org/abs/2307.08691)
-   [“FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision”by Jay Shah et al. (ArXiv abs/2407.08608, 2024)](https://arxiv.org/abs/2407.08608)
-   [*Flash-Attention-4*by Tri Dao (Dao AI Research Lab, 2025)](https://github.com/Dao-AILab/flash-attention)
-   [“Imagen Video: High Definition Video Generation with Diffusion Models”by Jonathan Ho et al. (ArXiv abs/2210.02303, 2022)](https://arxiv.org/abs/2210.02303)
-   [“Language Models Are Few-Shot Learners”by Tom Brown et al. (ArXiv abs/2005.14165)](https://arxiv.org/abs/2005.14165)
-   [“Learning Transferable Visual Models from Natural Language Supervision”by Alec Radford et al. (International Conference on Machine Learning, 2021)](https://arxiv.org/abs/2103.00020)
-   [“Longformer: The Long-Document Transformer”by Iz Beltagy et al. (ArXiv abs/2004.05150, 2020)](https://arxiv.org/abs/2004.05150)
-   [“Mamba: Linear-Time Sequence Modeling with Selective State Spaces”by Albert Gu and Tri Dao (ArXiv abs/2312.00752, 2023)](https://arxiv.org/abs/2312.00752)
-   [“Matryoshka Representation Learning”by Aditya Kusupati et al. (Neural Information Processing Systems, 2022)](https://arxiv.org/abs/2205.13147)
-   [“Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”by Noam Shazeer et al. (ArXiv abs/1701.06538, 2017)](https://arxiv.org/abs/1701.06538)
-   [“Reformer: The Efficient Transformer”by Nikita Kitaev et al. (ArXiv abs/2001.04451, 2020)](https://arxiv.org/abs/2001.04451)
-   [“Robust Speech Recognition via Large-Scale Weak Supervision”by Alec Radford et al. (International Conference on Machine Learning, 2022)](https://arxiv.org/abs/2212.04356)
-   [“RoFormer: Enhanced Transformer with Rotary Position Embedding”by Jianlin Su et al. (ArXiv abs/2104.09864, 2021)](https://arxiv.org/abs/2104.09864)
-   [“SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis”by Dustin Podell et al. (ArXiv abs/2307.01952, 2023)](https://arxiv.org/abs/2307.01952)
-   [“Segment Anything”by Alexander Kirillov et al. (2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023)](https://arxiv.org/abs/2304.02643)
-   [“Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks”by Nils Reimers and Iryna Gurevych (ArXiv abs/1908.10084, 2019)](https://arxiv.org/abs/1908.10084)
-   [“The Llama 3 Herd of Models”by Aaron Grattafiori et al. (ArXiv 2407.21783, 2024)](https://arxiv.org/abs/2407.21783)
-   [“Video Diffusion Models”by Jonathan Ho et al. (ArXiv abs/2204.03458, 2022)](https://arxiv.org/abs/2204.03458)
-   [“Visual Instruction Tuning”by Haotian Liu et al. (ArXiv abs/2304.08485, 2023)](https://arxiv.org/abs/2304.08485)

## Developer Tools

Engines, frameworks, and the libraries you will actually import.

-   [*BitsAndBytes by bitsandbytes-foundation*](https://github.com/TimDettmers/bitsandbytes)
-   [*ComfyUI by comfyanonymous*](https://github.com/comfyanonymous/ComfyUI)
-   [“CUDA by Example: An Introduction to General Purpose GPU Programming”by Jason Sanders and Edward Kandrot (NVIDIA developer, 2025)](https://developer.nvidia.com/cuda-example)
-   [“CUDA C++ Programming Guide Release 13.0”by NVIDIA (2025)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
-   [“CUDA cuBLAS Release 13.0”by NVIDIA (2025)](https://docs.nvidia.com/cuda/cublas/)
-   [*CUTLASS by NVIDIA*](https://github.com/NVIDIA/cutlass)
-   [*DeepGEMM by DeepSeek-ai*](https://github.com/deepseek-ai/DeepGEMM)
-   [*Hugging Face Diffusers by Hugging Face*](https://huggingface.co/docs/diffusers/index)
-   [*LMCache*by LMCache Project](https://github.com/LMCache/LMCache)
-   [*NVIDIA Dynamo Documentation*by NVIDIA (2025)](https://docs.nvidia.com/dynamo/latest/index.html)
-   [*NVIDIA Nsight Systems by NVIDIA*](https://developer.nvidia.com/nsight-systems)
-   [*NVIDIA Triton Inference Server by NVIDIA*](https://github.com/triton-inference-server/server)
-   [*ONNX Runtime by Microsoft*](https://onnxruntime.ai/)
-   [“PyTorch Performance Tuning Guide”by Szymon Migacz (PyTorch Foundation, 2020)](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
-   [“PyTorch Profiler”by Shivam Raikundalia (PyTorch Foundation, 2021)](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
-   [*SGLang Project*by LMSYS Org](https://github.com/sgl-project/sglang)
-   [“TensorRT Documentation”by NVIDIA (2025)](https://docs.nvidia.com/deeplearning/tensorrt/)
-   [*TensorRT-LLM*by NVIDIA](https://github.com/NVIDIA/TensorRT-LLM)
-   [*Transformers*by Hugging Face](https://huggingface.co/docs/transformers/index)
-   [*vLLM Project by The Linux Foundation*](https://github.com/vllm-project/vllm)

## Frontier Open Models

Model cards and technical reports for the open weights worth serving.

-   [*DeepSeek*by DeepSeek AI](https://huggingface.co/deepseek-ai)
-   [*FLUX*by Black Forest Labs](https://huggingface.co/black-forest-labs)
-   [*Gemma*by Google](https://huggingface.co/google)
-   [*GLM*by Z.ai](https://huggingface.co/zai-org)
-   [*GPT OSS*by OpenAI](https://huggingface.co/openai)
-   [*Kimi*by Moonshot AI](https://huggingface.co/moonshotai)
-   [*Llama*by Meta Llama](https://huggingface.co/meta-llama)
-   [*MiniMax*by MiniMax AI](https://huggingface.co/MiniMaxAI)
-   [*Mistral*by Mistral AI](https://huggingface.co/mistralai)
-   [*Nemotron*by NVIDIA](https://huggingface.co/nvidia)
-   [*Orpheus*by Canopy Labs](https://huggingface.co/canopylabs)
-   [*Qwen*by Alibaba Qwen](https://huggingface.co/Qwen)
-   [*Wan*by Wan-AI](https://huggingface.co/Wan-AI)
-   [*Whisper*by OpenAI](https://huggingface.co/openai)

## GPU Infrastructure

Hardware documentation, architecture whitepapers, and interconnect.

-   [*Designing Data-Intensive Applications*by Martin Kleppmann (O’Reilly Media, 2017)](https://dataintensive.net/)
-   [*Grace Hopper / Grace Blackwell Systems*by NVIDIA](https://www.nvidia.com/en-us/data-center/grace-cpu/)
-   [*GPU Glossary*by Frye et al. (Modal, 2025)](https://modal.com/gpu-glossary)
-   [*InfiniBand*by NVIDIA](https://www.nvidia.com/en-us/networking/products/infiniband/)
-   [*Kubernetes Documentation*by The Kubernetes Authors (The Linux Foundation, 2025)](https://kubernetes.io/docs/home/)
-   [*NVIDIA Blackwell Architecture Technical Brief: Built for the Age of AI Reasoning*by NVIDIA (2025)](https://resources.nvidia.com/en-us-blackwell-architecture?ncid=no-ncid)
-   [*NVIDIA H100 Tensor Core GPU Architecture: Exceptional Performance, Scalability and Security for the Data Center*by NVIDIA (2023)](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)
-   [“NVIDIA Tesla: A Unified Graphics and Computing Architecture”by E. Lindholm et al. (IEEE Micro, March–April 2008)](https://ieeexplore.ieee.org/document/4523358)
-   [*NVLink / NVSwitch*by NVIDIA](https://www.nvidia.com/en-us/data-center/nvlink/)
-   [*Programming Massively Parallel Processors: A Hands-on Approach*by Wen-mei Hwu, David Kirk, Izzat El Hajj (Morgan Kaufmann, 2022)](https://www.goodreads.com/work/editions/10244675-programming-massively-parallel-processors-a-hands-on-approach)
-   [*SemiAnalysis*by Dylan Patel (SemiAnalysis, 2025)](https://semianalysis.com/)
-   [*Site Reliability Engineering: How Google Runs Production Services*edited by Betsy Beyer et al. (O’Reilly Media, 2017)](https://sre.google/books/)

## Inference Optimization Research

Where the techniques in Chapter 5 come from: FlashAttention, speculation, quantization, paged attention.

-   [“Adversarial Diffusion Distillation”by Axel Sauer et al. (European Conference on Computer Vision, 2023)](https://arxiv.org/abs/2311.17042)
-   [“Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation”by Ofir Press, Noah Smith, and Mike Lewis (ArXiv abs/2108.12409, 2021)](https://arxiv.org/abs/2108.12409)
-   [“AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration”by Song Han (MIT, 2024)](https://github.com/mit-han-lab/llm-awq)
-   [*Cache-DIT*by Vipshop](https://github.com/vipshop/cache-dit)
-   [“CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion”by Jiayi Yao et al. (Proceedings of the Twentieth European Conference on Computer Systems, 2024)](https://arxiv.org/abs/2405.16444)
-   [“Adding Conditional Control to Text-to-Image Diffusion Models”by Lvmin Zhang et al. (International Conference on Computer Vision, 2023)](https://arxiv.org/abs/2302.05543)
-   [“Beyond the Buzz: A Pragmatic Take on Inference Disaggregation”by Tiyasa Mitra et al. (ArXiv abs/2506.05508, 2025)](https://arxiv.org/abs/2506.05508)
-   [“Break the Sequential Dependency of LLM Inference Using Lookahead Decoding”by Yichao Fu et al. (ArXiv abs/2402.02057, 2024)](https://arxiv.org/abs/2402.02057)
-   [“EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty”by Yuhui Li et al. (ArXiv abs/2401.15077, 2024)](https://arxiv.org/abs/2401.15077)
-   [“EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees”by Yuhui Li et al. (Conference on Empirical Methods in Natural Language Processing, 2024)](https://arxiv.org/abs/2406.16858)
-   [“EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test”by Yuhui Li et al. (ArXiv abs/2503.01840, 2025)](https://arxiv.org/abs/2503.01840)
-   [“FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving”by Ye et al. (ArXiv abs/2501.01005, 2025)](https://arxiv.org/abs/2501.01005)
-   [“GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers”by Elias Frantar (ArXiv abs/2210.17323, 2022)](https://arxiv.org/abs/2210.17323)
-   [“High-Resolution Image Synthesis with Latent Diffusion Models”by Robin Rombach et al. (Conference on Computer Vision and Pattern Recognition (CVPR), 2021)](https://arxiv.org/abs/2112.10752)
-   [“Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference”by Simian Luo (ArXiv abs/2310.04378, 2023)](https://arxiv.org/abs/2310.04378)
-   [“LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale”by Tim Dettmers et al. (ArXiv abs/2208.07339, 2022)](https://arxiv.org/abs/2208.07339)
-   [“Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads”by Tianle Cai et al. (ArXiv abs/2401.10774, 2024)](https://arxiv.org/abs/2401.10774)
-   [“Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism”by Mohammad Shoeybi et al. (ArXiv abs/1909.08053, 2019)](https://arxiv.org/abs/1909.08053)
-   [“Efficient Memory Management for Large Language Model Serving with PagedAttention”by Woosuk Kwon et al. (Proceedings of the 29th Symposium on Operating Systems Principles, 2023)](https://arxiv.org/abs/2309.06180)
-   [“Fast Inference from Transformers via Speculative Decoding”by Yaniv Leviathan et al. (International Conference on Machine Learning, 2022)](https://arxiv.org/abs/2211.17192)
-   [“Ring Attention with Blockwise Transformers for Near-Infinite Context”by Hao Liu et al. (ArXiv abs/2310.01889, 2023)](https://arxiv.org/abs/2310.01889)
-   [“SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration”by Jintao Zhang et al. (ArXiv abs/2410.02367, 2024)](https://arxiv.org/abs/2410.02367)
-   [*Sequence/Context Parallelism*by Megatron-LM for NVIDIA](https://github.com/NVIDIA/Megatron-LM)
-   [*SmoothQuant*by Song Han (MIT)](https://github.com/mit-han-lab/smoothquant)
-   [“SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot”by Elias Frantar and Dan Alistarh, (ArXiv abs/2301.00774, 2023)](https://arxiv.org/abs/2301.00774)
-   [“SpecVLM: Fast Speculative Decoding in Vision-Language Models”Haiduo Huang et al. (ArXiv abs/2509.11815, 2025)](https://arxiv.org/abs/2509.11815)
-   [“TeaCache: Timestep Embedding Aware Cache”by Feng Liu et al. (Alibaba TongYi Vision Intelligence Lab, ArXiv abs/2411.19108, 2025)](https://github.com/ali-vilab/TeaCache)

## Intelligence Evaluation

Benchmarks, their construction, and their well-documented limits.

-   [*ARC AGI Prize*by Greg Kamradt (2025)](https://arcprize.org/)
-   [*Evals for AI Engineers: Systematically Measuring and Improving AI Applications*by Shreya Shankar and Hamel Husain (O’Reilly Media, forthcoming 2026)](https://www.oreilly.com/library/view/evals-for-ai/9798341660717/)
-   [*Grade School Math: Training Verifiers to Solve Math Word Problems*by Karl Cobbe and Vineet Kosaraju (ArXiv abs/2110.14168, 2021)](https://github.com/openai/grade-school-math)
-   [“How to Fine-Tune Qwen3 to GPT-4o Level Performance”by Greg Schoeninger (Fine-Tune Fridays, Oxen AI, 2025)](https://ghost.oxen.ai/how-to-fine-tune-qwen3-to-gpt-4o-level-performance/)
-   [“Humanity’s Last Exam”by Long Phan et al. (Center for AI Safety and Scale AI, ArXiv abs/2501.14249, 2025)](https://www.lastexam.ai/)
-   [“HumanEval: Evaluating Large Language Models Trained on Code”by Mark Chen et al. (OpenAI, 2021)](https://github.com/openai/human-eval)
-   [“MMLU: Measuring Massive Multitask Language Understanding”by Dan Hendrycks et al. (Proceedings of the International Conference on Learning Representations (ICLR), 2021)](https://github.com/hendrycks/test)
-   [“MTEB: Massive Text Embedding Benchmark”by Niklas Muennighoff et al. (Conference of the European Chapter of the Association for Computational Linguistics, 2022)](https://arxiv.org/abs/2210.07316)
-   [“SWE-Bench: Can Language Models Resolve Real-World Github Issues?”by Carlos Jimenez et al. (Proceedings of the International Conference on Learning Representations (ICLR), 2024)](https://www.swebench.com/)

109 sources

Reproduced from Appendix B of *Inference Engineering*. Links go to the primary sources; anything worth understanding properly is better read there than summarized here.

---

# Software

<!-- https://learn-inference.com/chapters/software -->

NVIDIA’s lead in inference is not just about silicon. It is about the mature software stack sitting on top of it, and the fact that everyone’s kernels are written against CUDA.

Hardware iterates on a two-year cycle. Software iterates weekly, and it is common to need a nightly build of half your dependency tree just to serve a model released that morning. That asymmetry is why the software layer, not the hardware layer, is where most inference engineering actually happens.

A handful of organizations carry most of the weight. NVIDIA invests heavily from CUDA up through Dynamo. Hugging Face runs the model registry and maintains Transformers and Diffusers. The Linux Foundation hosts PyTorch and vLLM. LMSYS builds SGLang. Beneath them are thousands of individual open-source contributors.

This chapter climbs the stack. Each layer trades control for productivity, and knowing what the layer below yours is doing is what lets you diagnose the problems that leak upward.

## 5 sections

1.  [4.1CUDACUDA kernels for inference · CUDA kernel selection · Reducing memory accesses with kernel fusion](https://learn-inference.com/chapters/software/cuda)
2.  [4.2Deep learning frameworks and librariesPyTorch · Model file formats · ONNX Runtime and TensorRT · Transformers and Diffusers](https://learn-inference.com/chapters/software/frameworks)
3.  [4.3Inference enginesvLLM · SGLang · TensorRT-LLM](https://learn-inference.com/chapters/software/engines)
4.  [4.4NVIDIA Dynamo](https://learn-inference.com/chapters/software/dynamo)
5.  [4.5Performance benchmarking and load testingPerformance benchmarking tooling · Performance benchmarking tips · Profiling performance](https://learn-inference.com/chapters/software/benchmarking)

---

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

---

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

---

# NVIDIA Dynamo · Software

<!-- https://learn-inference.com/chapters/software/dynamo -->

Dynamo sits above the engines. Where an engine makes one model fast on one node, Dynamo coordinates many engine instances across many nodes, which is the [infrastructure layer](https://learn-inference.com/chapters/inference/scale-changes-problem) problem from Chapter 0, productized.

Its most interesting capability is dynamic [disaggregation](https://learn-inference.com/chapters/techniques/disaggregation): running prefill and decode on separate worker pools and shifting the ratio between them as traffic changes. Doing that by hand means picking a split and living with it; doing it dynamically means the fleet follows the workload.

It also handles KV-cache-aware routing, so requests land on the worker that already holds their prefix. That is a meaningful win at scale and pure overhead below it.

---

# Inference engines · Software

<!-- https://learn-inference.com/chapters/software/engines -->

Three engines are competitive: vLLM, SGLang, and TensorRT-LLM. All three give good out-of-the-box performance for LLMs and the architectures that resemble them, and all three support the techniques in [Chapter 5](https://learn-inference.com/chapters/techniques): continuous batching, post-training quantization, speculative decoding, prefix caching, parallelism, disaggregation.

Their value is configurability. Working with pre-optimized components means your time goes into testing combinations rather than reimplementing the same attention kernel badly.

> Figure. Choosing an inference engine by what you actually need. All three support the same core techniques. Turn on the constraints that apply to you and watch the field narrow. The choice is usually made by requirements rather than benchmarks.Constants from the book

### 4.3.1 vLLM

vLLM has the largest share by a distance: more GitHub stars than the other two combined. Released in mid-2023 out of UC Berkeley, it now lives under the PyTorch Project at the Linux Foundation.

Breadth is the selling point: NVIDIA, AMD, and Intel GPUs plus Google TPUs, and day-zero support for essentially every open model. vLLM Omni extends it to image, audio, and video.

The tradeoff follows the principle from [Chapter 1](https://learn-inference.com/chapters/prerequisites/scale-and-specialization): constraints buy performance. A platform that must serve every model on every accelerator gives up some of the ceiling a narrower engine can reach. Configured well it is fast; it is usually not the fastest.

### 4.3.2 SGLang

SGLang, from LMSYS, is close to vLLM on both performance and ease. Its distinguishing angle is pairing a fast backend with a flexible frontend language, so you can customize one component without rewriting the rest. It works day zero with labs like DeepSeek, Qwen, and Kimi, and its investment in large-scale MoE serving is why it is the engine of choice at xAI.

### 4.3.3 TensorRT-LLM

NVIDIA’s engine, and the fastest of the three when you can use it. Its edge is access: NVIDIA engineers write its kernels, some of them closed-source, tuned hard for Hopper and Blackwell. The older 0.x releases built literal TensorRT engines (an artifact per model, precision, and GPU), but the modern version is a standalone PyTorch-based package served much like the other two.

The cost is everything else. The learning curve is steep, model support is narrower, new architectures arrive later, it is NVIDIA-only, and it does not do image or video generation.

Which to pick

Learn all three and choose per deployment. Start with vLLM or SGLang, because getting to a working server quickly is worth more early than a percentage of throughput. Move to TensorRT-LLM when the workload is stable, the volume is high enough that the last increment of throughput is real money, and the model is one it supports well.

---

# Deep learning frameworks and libraries · Software

<!-- https://learn-inference.com/chapters/software/frameworks -->

### 4.2.1 PyTorch

PyTorch is where nearly every model is defined, trained, and first run. For inference the relevant machinery is `torch.compile`, which traces your model into a graph and hands it to a backend that fuses operations and selects kernels: much of [section 4.1](https://learn-inference.com/chapters/software/cuda), applied automatically.

Eager PyTorch is excellent for development and leaves real performance on the table in production. That gap is what inference engines exist to close.

### 4.2.2 Model file formats

Weights ship as files, and the format has security and performance consequences. Pickle-based `.bin` checkpoints can execute arbitrary code on load, which is why **safetensors** replaced them: a flat, memory-mappable layout that loads faster and cannot run code.

GGUF is the format of the local-inference world, built around quantized weights and CPU-plus-GPU splitting. Compiled engine formats like TensorRT plans are different again: hardware- and configuration-specific artifacts that are built, not downloaded, and that must be rebuilt when either changes.

### 4.2.3 ONNX Runtime and TensorRT

ONNX is an interchange format: export a model once, run it anywhere with an ONNX runtime. TensorRT is NVIDIA’s optimizing compiler, which takes a graph and produces a plan tuned for one specific GPU.

Both matter more outside LLM serving than within it. For transformers, the dedicated engines in [section 4.3](https://learn-inference.com/chapters/software/engines) have overtaken them; for image and video models, TensorRT remains an excellent answer.

### 4.2.4 Transformers and Diffusers

Hugging Face’s libraries are the reference implementations. When a new architecture lands, the Transformers implementation is usually the first correct one, and the engines follow.

Use them for development, evaluation, and anything unusual enough that no engine supports it. Do not use them as your production server: they are not built for it, and the performance difference is not subtle.

---

# Techniques

<!-- https://learn-inference.com/chapters/techniques -->

Five techniques, each adapted from recent research, and each a trade: speed bought with something else you may need.

Quantization gives up some quality, speculation extra compute, caching memory, parallelism communication overhead, and disaggregation operational complexity. Applying them well means knowing which of those you can afford to give up.

## 5 sections

1.  [5.1QuantizationNumber formats · Quantization approaches · Measuring quality impact](https://learn-inference.com/chapters/techniques/quantization)
2.  [5.2Speculative decodingDraft-target speculative decoding · Medusa · EAGLE · N-gram speculation and lookahead decoding](https://learn-inference.com/chapters/techniques/speculative-decoding)
3.  [5.3CachingPrefix caching and KV cache re-use · Where to store the KV cache · Cache-aware routing · Long context handling](https://learn-inference.com/chapters/techniques/caching)
4.  [5.4Model parallelismTensor parallelism for lower latency · Expert parallelism for higher throughput · Multi-node inference](https://learn-inference.com/chapters/techniques/parallelism)
5.  [5.5DisaggregationHow disaggregation works · When to use disaggregation · Dynamic disaggregation with NVIDIA Dynamo](https://learn-inference.com/chapters/techniques/disaggregation)

---

# Caching · Techniques

<!-- https://learn-inference.com/chapters/techniques/caching -->

The KV cache already exists to stop decode recomputing the whole sequence every token. Prefix caching extends that idea across requests: if two prompts share a prefix, they can share the attention state computed for it.

### 5.3.1 Prefix caching and KV cache re-use

The prerequisite is an exact match from the start of the sequence. Attention state at position N depends on every token before it, so a shared prefix must be identical and at the front.

That constraint has a design consequence worth internalizing: put the stable parts of your prompt first. A system prompt followed by a document followed by the user’s question caches well. Injecting a timestamp at the top of an otherwise-identical system prompt destroys the cache for every request.

> Figure 5.7. Prefix caching, and the routing problem underneath it. A shared prefix only has to be processed once. But the cache lives on one replica, so without cache-aware routing an eight-replica fleet finds it one time in eight, which is where most of the theoretical win quietly goes.Illustrative numbers

### 5.3.2 Where to store the KV cache

VRAM is fastest and scarcest, and it competes directly with batch size: every gigabyte of cache is a gigabyte not available for concurrent requests. Host RAM is far larger and much slower, though [Grace CPUs](https://learn-inference.com/chapters/hardware/generations) at 900 GB/s make the offload far more attractive than PCIe does. Beyond that, dedicated cache tiers on NVMe or over the network trade latency for capacity.

The decision is straightforward once framed correctly: fetching a cached prefix is worth it whenever the fetch is faster than recomputing prefill. For a long prompt that bar is low.

### 5.3.3 Cache-aware routing

A load balancer that does not know about caches will happily send a request with a warm prefix to a cold replica. The fix is routing on cache locality rather than on connection count, which turns a single-replica optimization into a fleet-wide one.

This is a good illustration of the layering in [Chapter 0](https://learn-inference.com/chapters/inference/three-layers): a runtime optimization whose value is capped by an infrastructure decision. NVIDIA Dynamo’s KV-aware router attacks this directly.

### 5.3.4 Long context handling

Long context is a cache problem before it is anything else. Cache size grows linearly with sequence length and with concurrency, so 128k context at batch 64 is a memory requirement that dwarfs the weights, as the calculator in [Chapter 2](https://learn-inference.com/chapters/models/llm-mechanics) makes uncomfortably clear.

The mitigations are cache quantization, offloading colder blocks to slower memory tiers, and architectural choices like sliding-window attention that bound the cache by construction. Each gives something up, and long context is one of the few places where you may have to accept a quality cost to make the workload possible at all.

---

# Disaggregation · Techniques

<!-- https://learn-inference.com/chapters/techniques/disaggregation -->

Prefill and decode want opposite machines. Prefill is compute-bound and processes a whole prompt at once. Decode is memory-bandwidth-bound and emits one token at a time. Running both on the same workers means neither is well served.

### 5.5.1 How disaggregation works

Split them into separate pools. Prefill workers take the prompt, build the KV cache, and hand it to a decode worker that generates the output. Each pool scales on its own signal and can run different hardware, even different parallelism.

The cost is the handoff. That KV cache has to cross the network, and it is not small, which is precisely why the interconnect discussion in [5.4](https://learn-inference.com/chapters/techniques/parallelism) matters here.

> Figure. Prefill and decode, split into their own pools. Each pool scales on its own signal and can run its own hardware, which is the entire benefit. The cost is the arrow in the middle: the KV cache for every in-flight request has to cross the network, and it is not small. Below real scale the extra hardware costs more than the specialisation saves.

### 5.5.2 When to use disaggregation

At scale, with real traffic, and ideally with a workload whose prefill-to-decode ratio is lopsided. Long prompts with short answers benefit most, because that is where a shared pool is most obviously mis-sized.

Below that it is a straightforward loss. You have added a network hop, a cache transfer, and two things to scale in exchange for efficiency you were not short of. And it introduces a new bottleneck to watch: prefill queue depth, which is the first thing to look at when a disaggregated deployment degrades.

### 5.5.3 Dynamic disaggregation with NVIDIA Dynamo

A fixed split is wrong most of the time, because traffic mix changes through the day. [Dynamo](https://learn-inference.com/chapters/software/dynamo) shifts workers between the pools as the ratio moves, so the fleet tracks the workload instead of a guess made at deployment.

NVIDIA is taking this further in hardware: the Rubin CPX is a chip built specifically for the prefill side, which is disaggregation promoted from a serving architecture into silicon.

These five techniques cover text. The next chapter is what changes when the model is not an LLM.

---

# Model parallelism · Techniques

<!-- https://learn-inference.com/chapters/techniques/parallelism -->

Sooner or later one GPU is not enough: the model does not fit, or it fits and is too slow. Splitting it across GPUs solves that and introduces communication, which is the new thing that can bottleneck you.

Keep the bandwidths from [Chapter 3](https://learn-inference.com/chapters/hardware/instances) in mind throughout. NVLink is roughly an order of magnitude faster than InfiniBand, so a split that is comfortable inside a node can fall apart across nodes.

### 5.4.1 Tensor parallelism for lower latency

Tensor parallelism splits individual weight matrices across GPUs. Every GPU holds a slice of every layer and works on every token, which means all of them contribute to each forward pass: the effective memory bandwidth is the sum across devices, and decode gets faster.

It also means an all-reduce at every layer boundary. That is fine over NVLink within a node and usually unacceptable over InfiniBand between nodes, which is why tensor parallelism is typically capped at the size of one node.

### 5.4.2 Expert parallelism for higher throughput

For [Mixture of Experts](https://learn-inference.com/chapters/models/llm-mechanics) models, expert parallelism places whole experts on individual GPUs. Tokens route to whichever GPU holds the expert they need.

Communication is a token-sized all-to-all rather than a layer-sized all-reduce, which is far less traffic. The catch is load balance: routing is data-dependent, so a popular expert makes its GPU the bottleneck while others idle.

### 5.4.3 Multi-node inference

Past eight GPUs you are crossing nodes, and the interconnect drops by an order of magnitude. The usual answer is hybrid: tensor parallelism within each node where NVLink is fast, and a coarser-grained split across nodes where it is not.

Reliability changes too. More nodes means more things that can fail, and a single-model deployment spanning several nodes fails as a unit.

---

# Quantization · Techniques

<!-- https://learn-inference.com/chapters/techniques/quantization -->

Models are trained and stored in a native format, usually BF16 or FP16. **Quantization** (Representing model weights, activations, or cache in a lower-precision number format than the one the model was trained in.) converts some part of that to fewer bits.

It pays twice, which is what makes it the highest-leverage technique here. Fewer bits means fewer bytes to move, which directly relieves the memory bandwidth bottleneck that governs [decode](https://learn-inference.com/chapters/models/bottlenecks). And low-precision tensor cores are faster, so the compute-bound phases speed up too. FP8 roughly halves the traffic and doubles the available FLOPS.

### 5.1.1 Number formats

FP64 is scientific-computing precision; FP32 occasionally shows up in training and almost never here. That leaves 16, 8, and 4-bit formats, and the differences between them are not just width.

A format has a precision, a type (integer or floating-point), and a scale factor mapping its values back to the higher-precision range. Together these determine two properties: **dynamic range**, the spread between the smallest and largest representable value, and **granularity**, how many values share a single scale factor.

Dynamic range is why floating-point beats integer for inference. Floating-point splits its bits into sign, exponent, and mantissa; the exponent is what lets it represent very large and very small values in the same format. That matters because outliers carry real signal in neural networks, and an integer format flattens them.

> Figure 5.1. Bit layout, dynamic range, and granularity. The exponent bits buy dynamic range, which is why floating-point beats integer for inference: outliers carry real signal. Granularity decides whether those outliers survive being quantized at all.Constants from the book

Granularity is the other half, and it is where the modern formats earn their keep. Quantizing per tensor is cheapest and smooths outliers into the average. Per channel is better. Per block of 32 (what MXFP8 and MXFP4 do) keeps outliers local enough to survive, at the cost of storing and applying many more scale factors. NVFP4 goes finer still: blocks of 16 under a secondary global scale factor.

This is why FP4 and NVFP4 behave so differently despite being the same width. Four bits gives you sixteen values, which is hopeless across a whole tensor and perfectly reasonable across a small block.

### 5.1.2 Quantization approaches

Two axes. **When**: post-training quantization converts an existing checkpoint and is what almost everyone does; quantization-aware training bakes it into the training run and needs access most teams do not have.

**What** matters more, because the risk is uneven across the model:

-   **Weights** quantize well. They are static, their distribution is known in advance, and FP8 weights are close to free quality-wise.
-   **Activations** are riskier. They vary with input and carry the outliers that matter, so they need more care and better granularity.
-   **KV cache** is riskier still. Errors here compound across every subsequent token rather than staying local.

The pragmatic order is weights first, then activations, then cache only if you need the memory badly enough to measure carefully.

### 5.1.3 Measuring quality impact

Quantization is the one technique in this chapter that can silently make your product worse. It produces no errors and delivers the promised throughput, while degrading output quality in ways that aggregate benchmarks average away.

Do not skip this

Run your own evaluations from [section 1.3.1](https://learn-inference.com/chapters/prerequisites/model-selection) against the unquantized baseline, on your own hardest cases. Public benchmark deltas will not catch a regression specific to your domain: a delta indistinguishable from noise can hide a real regression on the one task your product depends on.

---

# Speculative decoding · Techniques

<!-- https://learn-inference.com/chapters/techniques/speculative-decoding -->

Decode is stubbornly serial: one forward pass, one token. That is the constraint speculation attacks.

A cheap speculator proposes several draft tokens. The target model verifies all of them in a single forward pass (verification is parallel even though generation is not) and accepts the longest prefix that matches what it would have produced. Accept N drafts and you have produced N+1 tokens for the cost of roughly one target pass.

Crucially, output is identical to unspeculated decoding. This is not an approximation; rejected drafts are discarded.

> Figure 5.3. Speculative decoding, and where it stops paying. Drafts are cheap but not free, and one rejection discards everything after it. Push acceptance down or draft length up and the speedup goes below 1, at which point you are paying for compute to go slower.Illustrative numbers

The economics come down to three numbers: what a draft token costs, how often drafts are accepted, and how long the draft is. Long drafts look attractive and pay much less than expected, because a rejection discards everything after it: acceptance compounds geometrically.

One more caveat that the arithmetic above hides. Speculation spends compute to save time, which is a good trade when compute is idle. At high batch sizes it is not idle, and those draft passes compete with real requests. Speculation is a latency optimization for lightly-batched serving, and it can reduce total throughput.

### 5.2.1 Draft-target speculative decoding

The original form: a small model from the same family drafts for a large one, say a 1B drafting for a 70B. It works because the two share a tokenizer and training distribution, so the small model is right about easy tokens, which are most tokens.

The cost is a second model to deploy, version, and keep resident in VRAM.

### 5.2.2 Medusa

Medusa removes the second model. It bolts extra decoding heads onto the target itself, each predicting a token further ahead, so drafts come from the model that is already loaded.

No separate model to deploy or hold resident, in exchange for limits on how many drafts the heads can produce and how often they are accepted, and a training step to fit the heads.

### 5.2.3 EAGLE

EAGLE is the current state of the art and the one most production deployments reach for. Rather than predicting tokens from the output distribution, it takes the target model’s hidden states as input and predicts at the feature level.

That extra context makes the drafts substantially more accurate, and since acceptance rate is what the entire technique turns on, better drafts translate almost directly into speedup. EAGLE-2 adds dynamic draft trees, letting the speculator explore several candidate continuations rather than committing to one line, and EAGLE-3 scales the approach further.

### 5.2.4 N-gram speculation and lookahead decoding

The cheapest speculators involve no model at all. N-gram speculation keeps a dictionary mapping recent prefixes to likely suffixes and drafts from that.

It sounds too crude to work, and for open-ended chat it mostly is. For workloads with heavy verbatim repetition it is remarkable: code editing where most of a file is reproduced unchanged, summarization that quotes the source, structured output with fixed schemas. The draft costs a string lookup rather than a forward pass, and acceptance is high on exactly the workloads where a neural drafter would be overkill.

---

# Developers

<!-- https://learn-inference.com/developers -->

# Learn Inference for developers and agents

The chapter and section index behind this site is also a small public JSON API: no account, no API key, and no request quota to buy. It is read-only, so the live endpoints below are already safe to try.

## Quickstart

List every chapter:

```
curl https://learn-inference.com/api/v1/chapters
```

Get one chapter and its sections:

```
curl https://learn-inference.com/api/v1/chapters/inference
```

The full surface is described machine-readably at [/openapi.json](https://learn-inference.com/openapi.json) (OpenAPI 3.1), with typed request and response schemas an agent can load directly into a function-calling tool definition.

## MCP server

The same index is also an MCP server at `https://learn-inference.com/api/mcp`, for MCP-native clients like Claude Desktop or Claude Code. It exposes two tools: `list_chapters` and `get_chapter`, no authentication required.

```
{
  "mcpServers": {
    "learn-inference": {
      "url": "https://learn-inference.com/api/mcp"
    }
  }
}
```

It also publishes an [MCP server card](https://learn-inference.com/.well-known/mcp-server-card) for automatic discovery, per the draft [SEP-2127](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2127) proposal.

## Errors

Every non-2xx response, including an unknown path under `/api/` and an unsupported method on a real one, is JSON in the same shape: a stable `code`, a human-readable `message`, and usually a `hint` naming the next call to make.

```
curl https://learn-inference.com/api/v1/chapters/not-a-real-slug

{
  "error": {
    "code": "chapter_not_found",
    "message": "No chapter matches slug \"not-a-real-slug\".",
    "hint": "GET /api/v1/chapters for the list of valid slugs."
  }
}
```

## Versioning and rate limits

The API is versioned in the URL path (`/api/v1/...`). A breaking change ships under a new version prefix rather than changing this one in place; this version keeps working for at least 90 days after a new one ships, announced with a `Deprecation` response header and a `Sunset` date before removal.

Every response carries `RateLimit-Limit`, `RateLimit-Remaining`, and `RateLimit-Reset` headers. Going over the limit returns `429` with a `Retry-After` header, in the same JSON error shape as above.

## Full text

For reading rather than querying, every page is available as Markdown by appending `.md` to its URL, or by sending `Accept: text/markdown`. The whole book is one document at [/llms-full.txt](https://learn-inference.com/llms-full.txt), indexed at [/llms.txt](https://learn-inference.com/llms.txt).

## Reference

-   [OpenAPI 3.1 spec](https://learn-inference.com/openapi.json)
-   [GET /api/v1/chapters](https://learn-inference.com/api/v1/chapters)
-   [GET /api/v1/chapters/{slug}](https://learn-inference.com/api/v1/chapters/inference)
-   [MCP server](https://learn-inference.com/api/mcp)
-   [llms.txt](https://learn-inference.com/llms.txt)
-   [llms-full.txt](https://learn-inference.com/llms-full.txt)
-   [sitemap.xml](https://learn-inference.com/sitemap.xml)

---

# Privacy

<!-- https://learn-inference.com/privacy -->

# Privacy

Nothing here needs your name. No account, no email, nothing following you to the next site. This page is the actual mechanism, not a lawyer’s paraphrase of it.

## Ask AI

Type a question and it travels through Vercel’s AI Gateway, along with whatever page you’re reading, so the answer can be specific to it. It can only read pages already on this site. Not the web, not code execution, not the server’s files.

The server hashes your IP before anything else touches it. That hash stops one browser from hammering the endpoint, nothing else, and the raw address is never written down. Your side of the conversation lives in local storage on your machine. The server keeps a matching copy under the same session, so a reply that was mid-stream when you reloaded can keep going. Clear the conversation and your copy disappears.

## Everything else

Vercel Analytics and Speed Insights count visits and page-load speed in aggregate, no cookie involved. The API and MCP server need no key and rate-limit by IP in memory only, gone the moment the server restarts. Dark mode or light lives in local storage too. Fonts are baked into the site at build time, so even the first load never asks Google for anything.
