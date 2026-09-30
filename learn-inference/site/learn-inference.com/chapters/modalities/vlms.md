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
