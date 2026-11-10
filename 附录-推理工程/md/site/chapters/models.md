# Models

<!-- https://learn-inference.com/chapters/models -->

You cannot make something faster without knowing where its time goes. This chapter builds a model up from a matrix multiplication, then does the one calculation that tells you which resource you are actually short of.

The lineage is longer than the hype suggests. Perceptrons were built in hardware in the 1950s, abandoned, then revived as multi-layer networks once backpropagation gave them a way to learn. Depth arrived in the 2000s. AlexNet showed in 2012 that deep networks could do useful work and that GPUs were the way to run them.

The part that matters for us starts in 2017, when Vaswani and colleagues published “Attention Is All You Need.” A transformer is a neural network with an attention mechanism that learns relationships between positions in a sequence, and it turned out to generalize remarkably well. The models you will serve for text, images, audio, and video are nearly all transformer variants.

Two styles matter for inference, and they have opposite performance characteristics.

-   **Autoregressive generation** starts from a tokenized sequence and predicts the next token, repeatedly. LLMs work this way.
-   **Iterative denoising** starts from random noise and refines toward an output over a fixed number of steps. Image and video generation work this way.

## 5 sections

1.  [2.1Neural networksLinear layers and matmul · Activation functions](https://learn-inference.com/chapters/models/neural-networks)
2.  [2.2LLM inference mechanicsLLM architecture · Transformer blocks · Attention · Mixture of Experts models](https://learn-inference.com/chapters/models/llm-mechanics)
3.  [2.3Image generation inference mechanicsImage generation model architecture · Few-step image generation models · Video generation](https://learn-inference.com/chapters/models/image-mechanics)
4.  [2.4Calculating inference bottlenecksOps:byte ratio and arithmetic intensity · LLM inference bottlenecks · Image generation inference bottlenecks](https://learn-inference.com/chapters/models/bottlenecks)
5.  [2.5Optimizing attention](https://learn-inference.com/chapters/models/optimizing-attention)
