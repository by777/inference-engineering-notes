# Learn Inference

> An interactive guide to inference engineering: how generative AI models are served in production, from attention kernels to multi-cloud capacity.

An interactive companion to *Inference Engineering* by Philip Kiely (https://www.baseten.co/inference-engineering/). It follows the book's structure and covers the same ground, with the explanations rewritten and simulators in place of the parts that are easier to understand by turning a dial.

Every page is available as Markdown by appending `.md` to its URL. The whole book as one document is at https://learn-inference.com/llms-full.txt.

Figures are interactive simulators. In the Markdown they appear as a titled placeholder with a link, because what they teach is the response to input, which does not survive being written down.

## 0. Inference

The three layers you have to get right. Training teaches a model what it knows. Inference is everything that happens afterward, every time someone uses it. Serving a generative model in production means solving three problems at once: runtime, infrastructure, and tooling. None of them can be skipped.

- [Inference](https://learn-inference.com/chapters/inference): chapter overview
- [0.1 Two phases, two disciplines](https://learn-inference.com/chapters/inference/two-phases): Section 0.1.
- [0.2 The three layers](https://learn-inference.com/chapters/inference/three-layers): Section 0.2.
- [0.3 Six techniques that define the runtime](https://learn-inference.com/chapters/inference/runtime-techniques): Section 0.3.
- [0.4 Scale changes the problem](https://learn-inference.com/chapters/inference/scale-changes-problem): Section 0.4.
- [0.5 Where to put the abstraction](https://learn-inference.com/chapters/inference/abstraction): Section 0.5.
- [0.6 A map of what follows](https://learn-inference.com/chapters/inference/map): Section 0.6.

## 1. Prerequisites

The decisions you make before you optimize anything. Every inference optimization is an answer to a question about your product. Before touching a kernel, you need to know what you are building, which model you are serving, and what 'fast enough' means in numbers rather than adjectives.

- [Prerequisites](https://learn-inference.com/chapters/prerequisites): chapter overview
- [1.1 Scale and specialization](https://learn-inference.com/chapters/prerequisites/scale-and-specialization): Section 1.1.
- [1.2 About your app](https://learn-inference.com/chapters/prerequisites/about-your-app): Covers AI-native applications, Online versus offline, Consumer versus B2B.
- [1.3 Model selection](https://learn-inference.com/chapters/prerequisites/model-selection): Covers Model evaluation, Fine-tuning for domain-specific quality, Distillation.
- [1.4 Measuring latency and throughput](https://learn-inference.com/chapters/prerequisites/latency-throughput): Covers Latency percentiles, End-to-end metrics.

## 2. Models

Architecture, and where the time actually goes. At serving time, a model is a long chain of matrix multiplications. This chapter builds up from a linear layer to a transformer block to a diffusion pipeline, then does the arithmetic that tells you whether your bottleneck is memory bandwidth or compute: the calculation worth doing once to build the intuition.

- [Models](https://learn-inference.com/chapters/models): chapter overview
- [2.1 Neural networks](https://learn-inference.com/chapters/models/neural-networks): Covers Linear layers and matmul, Activation functions.
- [2.2 LLM inference mechanics](https://learn-inference.com/chapters/models/llm-mechanics): Covers LLM architecture, Transformer blocks, Attention, Mixture of Experts models.
- [2.3 Image generation inference mechanics](https://learn-inference.com/chapters/models/image-mechanics): Covers Image generation model architecture, Few-step image generation models, Video generation.
- [2.4 Calculating inference bottlenecks](https://learn-inference.com/chapters/models/bottlenecks): Covers Ops:byte ratio and arithmetic intensity, LLM inference bottlenecks, Image generation inference bottlenecks.
- [2.5 Optimizing attention](https://learn-inference.com/chapters/models/optimizing-attention): Section 2.5.

## 3. Hardware

Reading a GPU spec sheet like it means something. Compute, memory, and the bandwidth between them. This chapter takes apart a modern datacenter GPU, disambiguates NVIDIA's architecture names and SKUs, and covers what changes when a model no longer fits on one card.

- [Hardware](https://learn-inference.com/chapters/hardware): chapter overview
- [3.1 GPU architecture](https://learn-inference.com/chapters/hardware/gpu-architecture): Covers Compute, Memory and caches.
- [3.2 GPU architecture generations](https://learn-inference.com/chapters/hardware/generations): Covers Hopper GPUs, Ada Lovelace GPUs, Blackwell GPUs, Rubin GPUs, Grace and Vera CPUs.
- [3.3 Instances](https://learn-inference.com/chapters/hardware/instances): Covers Multi-GPU instances, Multi-instance GPUs.
- [3.4 Other datacenter accelerator options](https://learn-inference.com/chapters/hardware/other-accelerators): Section 3.4.
- [3.5 Local inference](https://learn-inference.com/chapters/hardware/local-inference): Covers Desktop inference, Mobile inference.

## 4. Software

From CUDA kernels to inference engines. The stack between a model file and a served token: CUDA, PyTorch, the model formats, and the three inference engines that most production deployments actually run on: vLLM, SGLang, and TensorRT-LLM.

- [Software](https://learn-inference.com/chapters/software): chapter overview
- [4.1 CUDA](https://learn-inference.com/chapters/software/cuda): Covers CUDA kernels for inference, CUDA kernel selection, Reducing memory accesses with kernel fusion.
- [4.2 Deep learning frameworks and libraries](https://learn-inference.com/chapters/software/frameworks): Covers PyTorch, Model file formats, ONNX Runtime and TensorRT, Transformers and Diffusers.
- [4.3 Inference engines](https://learn-inference.com/chapters/software/engines): Covers vLLM, SGLang, TensorRT-LLM.
- [4.4 NVIDIA Dynamo](https://learn-inference.com/chapters/software/dynamo): Section 4.4.
- [4.5 Performance benchmarking and load testing](https://learn-inference.com/chapters/software/benchmarking): Covers Performance benchmarking tooling, Performance benchmarking tips, Profiling performance.

## 5. Techniques

The optimizations that move the numbers. Quantization, speculative decoding, KV cache re-use, model parallelism, and disaggregation. Each one trades precision, memory, complexity, or hardware for latency or throughput. This chapter is about knowing which trade you are making.

- [Techniques](https://learn-inference.com/chapters/techniques): chapter overview
- [5.1 Quantization](https://learn-inference.com/chapters/techniques/quantization): Covers Number formats, Quantization approaches, Measuring quality impact.
- [5.2 Speculative decoding](https://learn-inference.com/chapters/techniques/speculative-decoding): Covers Draft-target speculative decoding, Medusa, EAGLE, N-gram speculation and lookahead decoding.
- [5.3 Caching](https://learn-inference.com/chapters/techniques/caching): Covers Prefix caching and KV cache re-use, Where to store the KV cache, Cache-aware routing, Long context handling.
- [5.4 Model parallelism](https://learn-inference.com/chapters/techniques/parallelism): Covers Tensor parallelism for lower latency, Expert parallelism for higher throughput, Multi-node inference.
- [5.5 Disaggregation](https://learn-inference.com/chapters/techniques/disaggregation): Covers How disaggregation works, When to use disaggregation, Dynamic disaggregation with NVIDIA Dynamo.

## 6. Modalities

Beyond text: vision, speech, embeddings, pixels. Most non-text models are transformer variants, which means the LLM toolkit mostly transfers. Image and video generation are the exception: different architecture, different bottlenecks, different optimizations.

- [Modalities](https://learn-inference.com/chapters/modalities): chapter overview
- [6.1 Vision language models](https://learn-inference.com/chapters/modalities/vlms): Covers Video processing for vision language models, Omni-modal models.
- [6.2 Embedding models](https://learn-inference.com/chapters/modalities/embeddings): Covers Embedding model architecture, Embedding model inference.
- [6.3 ASR models](https://learn-inference.com/chapters/modalities/asr): Covers Single-chunk latency optimization, Long file latency optimization, Diarization.
- [6.4 TTS models](https://learn-inference.com/chapters/modalities/tts): Covers Streaming real-time text to speech, Speech-to-speech models.
- [6.5 Image generation models](https://learn-inference.com/chapters/modalities/image-models): Covers Image generation kernel optimization, One weird trick for faster image generation.
- [6.6 Video generation models](https://learn-inference.com/chapters/modalities/video-models): Covers Attention optimization and quantization, Context parallelism.

## 7. Production

The infrastructure around the model. Containers, autoscaling, cold starts, multi-cloud capacity, zero-downtime deploys, and the client code on the other end. A perfectly optimized runtime still falls over if the infrastructure around it cannot get GPUs when traffic arrives.

- [Production](https://learn-inference.com/chapters/production): chapter overview
- [7.1 Containerization](https://learn-inference.com/chapters/production/containerization): Covers Dependency management, NIMs.
- [7.2 Autoscaling](https://learn-inference.com/chapters/production/autoscaling): Covers Concurrency and batch sizing, Cold starts, Routing, load balancing, and queueing, Scale to zero, Independent component scaling.
- [7.3 Multi-cloud capacity management](https://learn-inference.com/chapters/production/multi-cloud): Covers GPU procurement, Geo-aware load balancing, Building for reliability, Security and compliance.
- [7.4 Testing and deployment](https://learn-inference.com/chapters/production/testing-deployment): Covers Zero-downtime deployment, Cost estimation, Observability.
- [7.5 Client code](https://learn-inference.com/chapters/production/client-code): Covers Client latency overhead, Asynchronous inference, Streaming and protocol support.
- [7.6 Where this leaves you](https://learn-inference.com/chapters/production/closing): Section 7.6.

## Glossary

Every term, defined once. A searchable reference for the vocabulary of inference engineering, from arithmetic intensity to voice activity detection.

- [Glossary](https://learn-inference.com/chapters/glossary): chapter overview

## Further reading

The papers and posts worth your time. Primary sources grouped by area: architecture, developer tools, frontier open models, GPU infrastructure, optimization research, and evaluation.

- [Further reading](https://learn-inference.com/chapters/reading): chapter overview
- [B.1 Architecture](https://learn-inference.com/chapters/reading#architecture): Section B.1.
- [B.2 Developer tools](https://learn-inference.com/chapters/reading#developer-tools): Section B.2.
- [B.3 Frontier open models](https://learn-inference.com/chapters/reading#frontier-models): Section B.3.
- [B.4 GPU infrastructure](https://learn-inference.com/chapters/reading#gpu-infrastructure): Section B.4.
- [B.5 Inference optimization research](https://learn-inference.com/chapters/reading#optimization-research): Section B.5.
- [B.6 Intelligence evaluation](https://learn-inference.com/chapters/reading#evaluation): Section B.6.

## Reference

- [Glossary](https://learn-inference.com/chapters/glossary): every term of art, defined once
- [Further reading](https://learn-inference.com/chapters/reading): the papers and posts behind the book
- [The source book](https://www.baseten.co/inference-engineering/): Inference Engineering by Philip Kiely
- [About](https://learn-inference.com/about): what this site is and how it relates to the book
- [Privacy](https://learn-inference.com/privacy): what is collected, what is sent to Groq, and what is not

## For developers and agents

Learn Inference also publishes its chapter and section index as JSON, for programs that would rather not scrape HTML or Markdown. The API is versioned, rate-limited, keyless, and its errors are JSON: see the developer page for the details.

- [Developer resources](https://learn-inference.com/developers): quickstart, errors, versioning, rate limits
- [OpenAPI spec](https://learn-inference.com/openapi.json): the JSON API described machine-readably
- [GET /api/v1/chapters](https://learn-inference.com/api/v1/chapters): every chapter, as JSON
- [GET /api/v1/chapters/{slug}](https://learn-inference.com/api/v1/chapters/inference): one chapter and its sections, as JSON
- [MCP server](https://learn-inference.com/api/mcp): the same chapter index as MCP tools, for MCP-native clients
