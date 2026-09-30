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
