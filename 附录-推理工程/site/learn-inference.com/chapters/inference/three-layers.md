# The three layers · Inference

<!-- https://learn-inference.com/chapters/inference/three-layers -->

Doing this well means solving three separate problems, and they do not substitute for one another. A team can be excellent at one and still have a service nobody wants to depend on.

> Figure 0.1. The three layers of an inference stack. Each layer owns a different problem, and a team can be excellent at one while another quietly fails. The layers are ordered by distance from the hardware: the runtime sits closest to the GPU, the tooling closest to the engineer.

The **runtime** is the narrowest of the three and the easiest to reason about: one model, one instance, and the question of how much useful work you extract from the GPU you are already paying for. It rests on a deep software stack that runs from CUDA up through PyTorch to inference engines like vLLM, SGLang, and TensorRT-LLM. Low-level work matters here in a way that surprises people coming from web services; a better attention kernel is worth more than most application-level cleverness.

The **infrastructure** layer exists because no single instance is enough. Whatever you do to one server, it will eventually receive more traffic than it can serve, and that is not a CUDA or PyTorch problem. It is a distributed-systems problem: replication, routing, and scaling.

The **tooling** layer is the one teams skip, because it feels like overhead next to the other two. It is the interface everything else is operated through, and its absence shows up as a team that cannot deploy without a specific person being awake.
