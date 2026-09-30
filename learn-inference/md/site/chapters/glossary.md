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
