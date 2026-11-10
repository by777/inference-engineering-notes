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
