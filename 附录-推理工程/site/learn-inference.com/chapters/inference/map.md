# A map of what follows · Inference

<!-- https://learn-inference.com/chapters/inference/map -->

The rest of the book works up the stack. Each chapter assumes the previous ones but stands on its own well enough to be read out of order if you came here for something specific.

-   **[Chapter 1, Prerequisites](https://learn-inference.com/chapters/prerequisites):** Every inference optimization is an answer to a question about your product. Before touching a kernel, you need to know what you are building, which model you are serving, and what 'fast enough' means in numbers rather than adjectives.
-   **[Chapter 2, Models](https://learn-inference.com/chapters/models):** At serving time, a model is a long chain of matrix multiplications. This chapter builds up from a linear layer to a transformer block to a diffusion pipeline, then does the arithmetic that tells you whether your bottleneck is memory bandwidth or compute: the calculation worth doing once to build the intuition.
-   **[Chapter 3, Hardware](https://learn-inference.com/chapters/hardware):** Compute, memory, and the bandwidth between them. This chapter takes apart a modern datacenter GPU, disambiguates NVIDIA's architecture names and SKUs, and covers what changes when a model no longer fits on one card.
-   **[Chapter 4, Software](https://learn-inference.com/chapters/software):** The stack between a model file and a served token: CUDA, PyTorch, the model formats, and the three inference engines that most production deployments actually run on: vLLM, SGLang, and TensorRT-LLM.
-   **[Chapter 5, Techniques](https://learn-inference.com/chapters/techniques):** Quantization, speculative decoding, KV cache re-use, model parallelism, and disaggregation. Each one trades precision, memory, complexity, or hardware for latency or throughput. This chapter is about knowing which trade you are making.
-   **[Chapter 6, Modalities](https://learn-inference.com/chapters/modalities):** Most non-text models are transformer variants, which means the LLM toolkit mostly transfers. Image and video generation are the exception: different architecture, different bottlenecks, different optimizations.
-   **[Chapter 7, Production](https://learn-inference.com/chapters/production):** Containers, autoscaling, cold starts, multi-cloud capacity, zero-downtime deploys, and the client code on the other end. A perfectly optimized runtime still falls over if the infrastructure around it cannot get GPUs when traffic arrives.

Two appendices close it out: a [glossary](https://learn-inference.com/chapters/glossary) of the vocabulary, and a set of [primary sources](https://learn-inference.com/chapters/reading) worth reading directly.

On knowledge cutoffs

Books have them too. The source material was finished in January 2026, and specific hardware, version numbers, and benchmark figures will drift. The underlying constraints move much more slowly. Memory bandwidth is the binding limit on decode today, and nothing about that is about to change.
