# Disaggregation · Techniques

<!-- https://learn-inference.com/chapters/techniques/disaggregation -->

Prefill and decode want opposite machines. Prefill is compute-bound and processes a whole prompt at once. Decode is memory-bandwidth-bound and emits one token at a time. Running both on the same workers means neither is well served.

### 5.5.1 How disaggregation works

Split them into separate pools. Prefill workers take the prompt, build the KV cache, and hand it to a decode worker that generates the output. Each pool scales on its own signal and can run different hardware, even different parallelism.

The cost is the handoff. That KV cache has to cross the network, and it is not small, which is precisely why the interconnect discussion in [5.4](https://learn-inference.com/chapters/techniques/parallelism) matters here.

> Figure. Prefill and decode, split into their own pools. Each pool scales on its own signal and can run its own hardware, which is the entire benefit. The cost is the arrow in the middle: the KV cache for every in-flight request has to cross the network, and it is not small. Below real scale the extra hardware costs more than the specialisation saves.

### 5.5.2 When to use disaggregation

At scale, with real traffic, and ideally with a workload whose prefill-to-decode ratio is lopsided. Long prompts with short answers benefit most, because that is where a shared pool is most obviously mis-sized.

Below that it is a straightforward loss. You have added a network hop, a cache transfer, and two things to scale in exchange for efficiency you were not short of. And it introduces a new bottleneck to watch: prefill queue depth, which is the first thing to look at when a disaggregated deployment degrades.

### 5.5.3 Dynamic disaggregation with NVIDIA Dynamo

A fixed split is wrong most of the time, because traffic mix changes through the day. [Dynamo](https://learn-inference.com/chapters/software/dynamo) shifts workers between the pools as the ratio moves, so the fleet tracks the workload instead of a guess made at deployment.

NVIDIA is taking this further in hardware: the Rubin CPX is a chip built specifically for the prefill side, which is disaggregation promoted from a serving architecture into silicon.

These five techniques cover text. The next chapter is what changes when the model is not an LLM.
