# Instances · Hardware

<!-- https://learn-inference.com/chapters/hardware/instances -->

You do not rent a GPU. You rent an instance: a virtual machine that happens to include one or more GPUs, plus host CPUs, host memory, storage, networking, and interconnect. Every one of those can be the thing that ruins your latency.

This is where cloud providers diverge. NVIDIA publishes reference architectures; providers build to their own preferences. Even the same GPU differs: an A100 comes in PCIe and SXM form factors, and SXM has about five percent higher memory bandwidth, enough that most A100 inference runs on it. Know exactly what you are provisioning, because a slow disk or a thin network link will bottleneck a perfectly good GPU.

### 3.3.1 Multi-GPU instances

The standard unit is a node of eight GPUs, connected by **NVLink** (NVIDIA's direct GPU-to-GPU interconnect. Up to 900 GB/s on Hopper and 1,800 GB/s on Blackwell.) for point-to-point traffic and NVSwitch for all-to-all coordination. Those links are what make it practical to split one model across eight cards.

Past eight, you need node-to-node interconnect, and the standard is InfiniBand at up to 400 Gb/s per NIC. Note the units: NVLink is measured in gigabytes per second, InfiniBand in gigabits. NVLink is roughly an order of magnitude faster.

That ratio governs every parallelism decision in [Chapter 5](https://learn-inference.com/chapters/techniques/parallelism). A strategy that saturates InfiniBand may be comfortable inside a node. Not every provider offers InfiniBand, and some offer it on only part of their fleet, so check before you design around it.

At the top end, NVIDIA sells rack-scale systems: the GB200 NVL72 puts 72 Blackwell GPUs and 36 Grace CPUs under one NVLink domain, with the Vera Rubin NVL144 CPX to follow. These exist to serve the largest models at serious traffic.

### 3.3.2 Multi-instance GPUs

The opposite problem is more common than people expect: the GPU is too big for the model. A three-billion parameter model cannot keep an H100 busy even at generous batch sizes, and the waste is pure margin loss.

Multi-instance GPU splits an A100, H100, H200, or B200 into as many as seven slices, each with its own compute, memory, and a proportional share of CPU, RAM, storage, and network. An H100 has eight memory slices and seven compute slices, which is why the partitioning looks strange to anyone expecting powers of two.

This is usually the better answer than reaching for an older, cheaper card. A fraction of a modern GPU carries modern architecture features and modern kernel support; an equivalently priced older card does not.
