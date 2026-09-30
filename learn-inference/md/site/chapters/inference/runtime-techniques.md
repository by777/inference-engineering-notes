# Six techniques that define the runtime · Inference

<!-- https://learn-inference.com/chapters/inference/runtime-techniques -->

The runtime layer is largely an exercise in applying a handful of techniques, most of them adapted from research published in the last few years. They come up constantly, they compose with each other, and each gets a proper treatment later in the book. It is worth meeting them now, because the rest of the material is organized around them.

What is easy to miss on a first pass is that none of these is free. Every one takes something away in exchange for what it gives, and choosing well means knowing which currency you are spending.

> Figure. Every technique is a purchase. Toggle techniques on the way you would configure a deployment, and the bill fills in: one line per cost, filed under the currency it spends. Turn everything on and count the rows: complexity is billed by four of the six techniques, and quality exactly once, by quantization. No combination of switches leaves the bill empty.

These apply well beyond text. Vision language models, embedding models, speech recognition, speech synthesis, image generation, and video generation all lean on the same set, with the details bent to fit each architecture. [Chapter 6](https://learn-inference.com/chapters/modalities) covers where the analogy holds and where it breaks.
