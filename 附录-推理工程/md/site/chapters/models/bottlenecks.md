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
