# NVIDIA Dynamo · Software

<!-- https://learn-inference.com/chapters/software/dynamo -->

Dynamo sits above the engines. Where an engine makes one model fast on one node, Dynamo coordinates many engine instances across many nodes, which is the [infrastructure layer](https://learn-inference.com/chapters/inference/scale-changes-problem) problem from Chapter 0, productized.

Its most interesting capability is dynamic [disaggregation](https://learn-inference.com/chapters/techniques/disaggregation): running prefill and decode on separate worker pools and shifting the ratio between them as traffic changes. Doing that by hand means picking a split and living with it; doing it dynamically means the fleet follows the workload.

It also handles KV-cache-aware routing, so requests land on the worker that already holds their prefix. That is a meaningful win at scale and pure overhead below it.
