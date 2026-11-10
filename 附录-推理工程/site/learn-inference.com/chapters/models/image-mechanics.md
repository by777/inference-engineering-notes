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
