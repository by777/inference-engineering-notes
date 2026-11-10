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
