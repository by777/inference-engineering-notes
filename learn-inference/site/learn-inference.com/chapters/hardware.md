# Hardware

<!-- https://learn-inference.com/chapters/hardware -->

A spec sheet has three numbers on it that matter, and which one you should care about depends entirely on what you are running. Reading them correctly is most of what hardware selection is.

Inference runs on accelerators, and in the datacenter that overwhelmingly means NVIDIA GPUs. There are three broad classes: datacenter GPUs racked in standardized chassis, workstation GPUs for professional desktops, and consumer cards. Serving at scale means the first category, not because the silicon is categorically better but because those cards ship with high-bandwidth interconnects, arrive in predictable configurations, and exist by the million.

You are almost certainly renting them. Cloud is the default: hyperscalers like AWS and GCP, or neoclouds like CoreWeave and Nebius. On-premise and air-gapped deployments are real but belong to large enterprises and governments; everyone else wants the ability to add capacity without a procurement cycle.

## 5 sections

1.  [3.1GPU architectureCompute · Memory and caches](https://learn-inference.com/chapters/hardware/gpu-architecture)
2.  [3.2GPU architecture generationsHopper GPUs · Ada Lovelace GPUs · Blackwell GPUs · Rubin GPUs · Grace and Vera CPUs](https://learn-inference.com/chapters/hardware/generations)
3.  [3.3InstancesMulti-GPU instances · Multi-instance GPUs](https://learn-inference.com/chapters/hardware/instances)
4.  [3.4Other datacenter accelerator options](https://learn-inference.com/chapters/hardware/other-accelerators)
5.  [3.5Local inferenceDesktop inference · Mobile inference](https://learn-inference.com/chapters/hardware/local-inference)
