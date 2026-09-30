# Video generation models · Modalities

<!-- https://learn-inference.com/chapters/modalities/video-models -->

Video is diffusion with a time axis, at far higher cost. Attention now spans space and time together, so sequence lengths dwarf anything in text, and the quadratic term in attention becomes the dominant cost of the entire pipeline.

### 6.6.1 Attention optimization and quantization

Because attention dominates, the optimizations from [section 2.5](https://learn-inference.com/chapters/models/optimizing-attention) matter more here than anywhere else. Then comes a family of caching tricks with no LLM equivalent (reusing outputs across timesteps, or hidden states across layers), worth thirty to forty percent when quality holds.

Quantization is close to mandatory rather than optional. Dropping to lower-precision tensor cores doubles available FLOPS, and video generation needs every one of them.

### 6.6.2 Context parallelism

Video is the case where multi-GPU serves a single request rather than more users. Context parallelism splits the sequence itself across GPUs, each handling a slice of the space-time volume and exchanging what attention requires.

This inverts the usual economics. Elsewhere you add GPUs to serve more people; here you add GPUs so one request completes in a tolerable time, and the interconnect bandwidth from [Chapter 3](https://learn-inference.com/chapters/hardware/instances) is what decides whether that scales.

That covers the models. The last chapter is everything around them.
