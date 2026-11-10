# Embedding models · Modalities

<!-- https://learn-inference.com/chapters/modalities/embeddings -->

Embedding models turn a variable-length input into a fixed-length vector. They power search, retrieval, clustering, and recommendation, and they are the quiet workhorse of production AI, running in high-volume, low-latency deployments behind search and retrieval.

### 6.2.1 Embedding model architecture

Either a small BERT-style encoder (under a billion parameters) or a repurposed LLM of a few billion, with the LLM-based family now the more capable of the two. Either way the output is pooled across positions into a single vector.

Small and encoder-only changes everything about serving. There is no autoregressive loop, so no decode phase and no KV cache. One forward pass produces the entire output.

### 6.2.2 Embedding model inference

Which makes embeddings almost purely compute-bound and almost perfectly batchable. There is no memory-bandwidth wall to work around, because you are not re-reading the weights per token.

The optimizations are correspondingly different: large batches, sequence-length bucketing so short inputs are not padded to the longest in the batch, and often a [fractional GPU](https://learn-inference.com/chapters/hardware/instances), since a 300M-parameter model cannot keep an H100 busy.

A high-performance pipeline usually parallelizes tokenization on the CPU alongside GPU inference. For a model this small, CPU-side preprocessing is a genuine bottleneck rather than a rounding error.
