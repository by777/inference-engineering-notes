# Model parallelism · Techniques

<!-- https://learn-inference.com/chapters/techniques/parallelism -->

Sooner or later one GPU is not enough: the model does not fit, or it fits and is too slow. Splitting it across GPUs solves that and introduces communication, which is the new thing that can bottleneck you.

Keep the bandwidths from [Chapter 3](https://learn-inference.com/chapters/hardware/instances) in mind throughout. NVLink is roughly an order of magnitude faster than InfiniBand, so a split that is comfortable inside a node can fall apart across nodes.

### 5.4.1 Tensor parallelism for lower latency

Tensor parallelism splits individual weight matrices across GPUs. Every GPU holds a slice of every layer and works on every token, which means all of them contribute to each forward pass: the effective memory bandwidth is the sum across devices, and decode gets faster.

It also means an all-reduce at every layer boundary. That is fine over NVLink within a node and usually unacceptable over InfiniBand between nodes, which is why tensor parallelism is typically capped at the size of one node.

### 5.4.2 Expert parallelism for higher throughput

For [Mixture of Experts](https://learn-inference.com/chapters/models/llm-mechanics) models, expert parallelism places whole experts on individual GPUs. Tokens route to whichever GPU holds the expert they need.

Communication is a token-sized all-to-all rather than a layer-sized all-reduce, which is far less traffic. The catch is load balance: routing is data-dependent, so a popular expert makes its GPU the bottleneck while others idle.

### 5.4.3 Multi-node inference

Past eight GPUs you are crossing nodes, and the interconnect drops by an order of magnitude. The usual answer is hybrid: tensor parallelism within each node where NVLink is fast, and a coarser-grained split across nodes where it is not.

Reliability changes too. More nodes means more things that can fail, and a single-model deployment spanning several nodes fails as a unit.
