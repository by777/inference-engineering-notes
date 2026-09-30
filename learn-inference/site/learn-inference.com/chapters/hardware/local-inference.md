# Local inference · Hardware

<!-- https://learn-inference.com/chapters/hardware/local-inference -->

Not all inference happens in a datacenter, and the constraints invert when it does not. There is no batching across users, because there is one user. Memory is small and shared with everything else on the device. Power and thermals matter in a way they never do on a rack.

### 3.5.1 Desktop inference

Consumer cards and Apple Silicon both run useful models now. The interesting property of unified memory architectures is that the usual VRAM ceiling does not apply the same way: the constraint becomes bandwidth rather than capacity, which changes which models are actually pleasant to run.

Batch size one is the defining fact. Every arithmetic-intensity argument from [Chapter 2](https://learn-inference.com/chapters/models/bottlenecks) applies at its most extreme, so local inference is memory-bandwidth-bound almost by definition, and quantization stops being an optimization and becomes a precondition.

### 3.5.2 Mobile inference

Phones ship neural accelerators (Apple’s Neural Engine, Qualcomm’s Hexagon) alongside their CPUs and GPUs. The models are small, heavily quantized, and usually converted to a device-specific runtime rather than served from PyTorch.

What makes on-device worth the trouble is rarely speed. It is that the data never leaves the device, and that the feature works on a plane. Those are product arguments, and they are often decisive regardless of what the benchmarks say.

Hardware sets the ceiling on all of this. How close you get to it is a question about the software that drives the chip, which is where the book turns next.
