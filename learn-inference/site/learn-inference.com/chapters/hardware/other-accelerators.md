# Other datacenter accelerator options · Hardware

<!-- https://learn-inference.com/chapters/hardware/other-accelerators -->

NVIDIA’s dominance in inference is real but not total. AMD ships the MI350 with competitive specs on its own software stack, which has improved to the point where major engines run on it. Google’s TPUs serve enormous internal and Cloud workloads. AWS builds Inferentia and Trainium. Groq and Cerebras take different architectural approaches aimed at latency.

The gap is rarely the silicon. It is CUDA and the decade of kernels, libraries, and engine support built on top of it. Moving off NVIDIA means accepting that some optimization you want will not be available yet, and pricing that against whatever capacity or cost advantage prompted the move.
