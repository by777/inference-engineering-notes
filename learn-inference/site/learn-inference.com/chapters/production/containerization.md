# Containerization · Production

<!-- https://learn-inference.com/chapters/production/containerization -->

Inference ships as a container image: the engine, the CUDA libraries, the Python dependencies, and your serving code. That image is the unit of deployment, and its size directly sets one of the four terms in your cold start.

GPU images are enormous. CUDA and PyTorch alone run to several gigabytes before you add anything, and a naive image runs to many more. Most of those bytes are not strictly necessary.

### 7.1.1 Dependency management

This stack is unusually brittle about versions. CUDA, the driver, PyTorch, and the engine all have to agree, and the failure mode is frequently a runtime error deep inside a kernel launch rather than a clean import failure.

Pin everything, exactly, including transitive dependencies. The cost of a fully reproducible image is a slower upgrade path. The cost of an unpinned one is a deploy that works on Tuesday and does not on Wednesday for reasons nobody changed.

### 7.1.2 NIMs

NVIDIA Inference Microservices are prebuilt containers: a model, an optimized engine, and an API, packaged together and tuned for specific GPUs.

They are the abstraction trade from [Chapter 0](https://learn-inference.com/chapters/inference/abstraction) in container form. You get a working optimized deployment without doing the optimization, and you give up the ability to change anything the packager did not expose. Good for standard models on standard hardware; frustrating the moment you need something specific.
