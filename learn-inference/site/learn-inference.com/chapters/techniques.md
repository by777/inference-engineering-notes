# Techniques

<!-- https://learn-inference.com/chapters/techniques -->

Five techniques, each adapted from recent research, and each a trade: speed bought with something else you may need.

Quantization gives up some quality, speculation extra compute, caching memory, parallelism communication overhead, and disaggregation operational complexity. Applying them well means knowing which of those you can afford to give up.

## 5 sections

1.  [5.1QuantizationNumber formats · Quantization approaches · Measuring quality impact](https://learn-inference.com/chapters/techniques/quantization)
2.  [5.2Speculative decodingDraft-target speculative decoding · Medusa · EAGLE · N-gram speculation and lookahead decoding](https://learn-inference.com/chapters/techniques/speculative-decoding)
3.  [5.3CachingPrefix caching and KV cache re-use · Where to store the KV cache · Cache-aware routing · Long context handling](https://learn-inference.com/chapters/techniques/caching)
4.  [5.4Model parallelismTensor parallelism for lower latency · Expert parallelism for higher throughput · Multi-node inference](https://learn-inference.com/chapters/techniques/parallelism)
5.  [5.5DisaggregationHow disaggregation works · When to use disaggregation · Dynamic disaggregation with NVIDIA Dynamo](https://learn-inference.com/chapters/techniques/disaggregation)
