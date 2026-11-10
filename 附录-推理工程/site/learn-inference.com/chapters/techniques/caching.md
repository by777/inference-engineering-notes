# Caching · Techniques

<!-- https://learn-inference.com/chapters/techniques/caching -->

The KV cache already exists to stop decode recomputing the whole sequence every token. Prefix caching extends that idea across requests: if two prompts share a prefix, they can share the attention state computed for it.

### 5.3.1 Prefix caching and KV cache re-use

The prerequisite is an exact match from the start of the sequence. Attention state at position N depends on every token before it, so a shared prefix must be identical and at the front.

That constraint has a design consequence worth internalizing: put the stable parts of your prompt first. A system prompt followed by a document followed by the user’s question caches well. Injecting a timestamp at the top of an otherwise-identical system prompt destroys the cache for every request.

> Figure 5.7. Prefix caching, and the routing problem underneath it. A shared prefix only has to be processed once. But the cache lives on one replica, so without cache-aware routing an eight-replica fleet finds it one time in eight, which is where most of the theoretical win quietly goes.Illustrative numbers

### 5.3.2 Where to store the KV cache

VRAM is fastest and scarcest, and it competes directly with batch size: every gigabyte of cache is a gigabyte not available for concurrent requests. Host RAM is far larger and much slower, though [Grace CPUs](https://learn-inference.com/chapters/hardware/generations) at 900 GB/s make the offload far more attractive than PCIe does. Beyond that, dedicated cache tiers on NVMe or over the network trade latency for capacity.

The decision is straightforward once framed correctly: fetching a cached prefix is worth it whenever the fetch is faster than recomputing prefill. For a long prompt that bar is low.

### 5.3.3 Cache-aware routing

A load balancer that does not know about caches will happily send a request with a warm prefix to a cold replica. The fix is routing on cache locality rather than on connection count, which turns a single-replica optimization into a fleet-wide one.

This is a good illustration of the layering in [Chapter 0](https://learn-inference.com/chapters/inference/three-layers): a runtime optimization whose value is capped by an infrastructure decision. NVIDIA Dynamo’s KV-aware router attacks this directly.

### 5.3.4 Long context handling

Long context is a cache problem before it is anything else. Cache size grows linearly with sequence length and with concurrency, so 128k context at batch 64 is a memory requirement that dwarfs the weights, as the calculator in [Chapter 2](https://learn-inference.com/chapters/models/llm-mechanics) makes uncomfortably clear.

The mitigations are cache quantization, offloading colder blocks to slower memory tiers, and architectural choices like sliding-window attention that bound the cache by construction. Each gives something up, and long context is one of the few places where you may have to accept a quality cost to make the workload possible at all.
