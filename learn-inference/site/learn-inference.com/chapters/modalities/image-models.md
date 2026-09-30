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
