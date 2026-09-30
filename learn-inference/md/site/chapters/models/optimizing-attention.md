# Optimizing attention · Models

<!-- https://learn-inference.com/chapters/models/optimizing-attention -->

Attention is the most expensive operation in a transformer and the most studied. The naive implementation is three lines: score queries against keys, normalize, weight the values. Each line reads from memory, computes, and writes back.

The problem is the intermediate matrices. For a 4,096-token sequence the score matrix is 4,096 by 4,096 (about 32 MB in FP16, roughly a high-resolution RAW photograph), and the naive version writes it to memory and reads it back, twice, for every head in every layer. The arithmetic is not the expensive part. The round trips are.

FlashAttention’s insight was that you can avoid materializing that matrix at all. Tile the computation, keep tiles in on-chip SRAM, and compute the softmax incrementally so the full score matrix never touches HBM. Same mathematical result, a fraction of the memory traffic. It is the canonical example of the principle that reorganizing memory access beats optimizing arithmetic.

Beyond exact attention, several approaches trade fidelity for time:

-   **Grouped-query attention** shares key and value heads across multiple query heads, shrinking the KV cache several times over. You saw its effect in the calculator in [section 2.2](https://learn-inference.com/chapters/models/llm-mechanics).
-   **Sliding window attention** limits each token to a fixed window of neighbours, making cost linear in sequence length rather than quadratic.
-   **Sparse and gated attention** approximate the full computation by attending selectively, trading exactness for scaling.

These are architectural decisions made at training time, not knobs you turn at inference. They belong here because they explain why two models with identical parameter counts can have wildly different serving costs, and why [model selection](https://learn-inference.com/chapters/prerequisites/model-selection) is the highest-leverage performance decision you make.

The bottleneck is as much a fact about the hardware as about the model, and what that hardware actually offers is the subject of the next chapter.
