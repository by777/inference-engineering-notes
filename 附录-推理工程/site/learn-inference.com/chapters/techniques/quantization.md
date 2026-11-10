# Quantization · Techniques

<!-- https://learn-inference.com/chapters/techniques/quantization -->

Models are trained and stored in a native format, usually BF16 or FP16. **Quantization** (Representing model weights, activations, or cache in a lower-precision number format than the one the model was trained in.) converts some part of that to fewer bits.

It pays twice, which is what makes it the highest-leverage technique here. Fewer bits means fewer bytes to move, which directly relieves the memory bandwidth bottleneck that governs [decode](https://learn-inference.com/chapters/models/bottlenecks). And low-precision tensor cores are faster, so the compute-bound phases speed up too. FP8 roughly halves the traffic and doubles the available FLOPS.

### 5.1.1 Number formats

FP64 is scientific-computing precision; FP32 occasionally shows up in training and almost never here. That leaves 16, 8, and 4-bit formats, and the differences between them are not just width.

A format has a precision, a type (integer or floating-point), and a scale factor mapping its values back to the higher-precision range. Together these determine two properties: **dynamic range**, the spread between the smallest and largest representable value, and **granularity**, how many values share a single scale factor.

Dynamic range is why floating-point beats integer for inference. Floating-point splits its bits into sign, exponent, and mantissa; the exponent is what lets it represent very large and very small values in the same format. That matters because outliers carry real signal in neural networks, and an integer format flattens them.

> Figure 5.1. Bit layout, dynamic range, and granularity. The exponent bits buy dynamic range, which is why floating-point beats integer for inference: outliers carry real signal. Granularity decides whether those outliers survive being quantized at all.Constants from the book

Granularity is the other half, and it is where the modern formats earn their keep. Quantizing per tensor is cheapest and smooths outliers into the average. Per channel is better. Per block of 32 (what MXFP8 and MXFP4 do) keeps outliers local enough to survive, at the cost of storing and applying many more scale factors. NVFP4 goes finer still: blocks of 16 under a secondary global scale factor.

This is why FP4 and NVFP4 behave so differently despite being the same width. Four bits gives you sixteen values, which is hopeless across a whole tensor and perfectly reasonable across a small block.

### 5.1.2 Quantization approaches

Two axes. **When**: post-training quantization converts an existing checkpoint and is what almost everyone does; quantization-aware training bakes it into the training run and needs access most teams do not have.

**What** matters more, because the risk is uneven across the model:

-   **Weights** quantize well. They are static, their distribution is known in advance, and FP8 weights are close to free quality-wise.
-   **Activations** are riskier. They vary with input and carry the outliers that matter, so they need more care and better granularity.
-   **KV cache** is riskier still. Errors here compound across every subsequent token rather than staying local.

The pragmatic order is weights first, then activations, then cache only if you need the memory badly enough to measure carefully.

### 5.1.3 Measuring quality impact

Quantization is the one technique in this chapter that can silently make your product worse. It produces no errors and delivers the promised throughput, while degrading output quality in ways that aggregate benchmarks average away.

Do not skip this

Run your own evaluations from [section 1.3.1](https://learn-inference.com/chapters/prerequisites/model-selection) against the unquantized baseline, on your own hardest cases. Public benchmark deltas will not catch a regression specific to your domain: a delta indistinguishable from noise can hide a real regression on the one task your product depends on.
