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
