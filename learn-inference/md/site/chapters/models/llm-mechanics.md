# LLM inference mechanics · Models

<!-- https://learn-inference.com/chapters/models/llm-mechanics -->

Text does not enter a model as text. A tokenizer splits it into subword units (common words get one token, rarer ones get broken into pieces), and each becomes an integer index into a vocabulary. One token is roughly three-quarters of an English word.

Every request assembles into a single sequence: the input, an optional reasoning sequence for models that think before answering, and the output. Together these have to fit inside the model’s **context window** (The maximum number of tokens a model can process and generate for a single request, input and output combined.).

Getting from a chat request to that single sequence is the job of the chat template, which encodes roles, tool signatures, and multimodal inputs into the exact format the model saw during training. Templates differ subtly between models, and a template applied slightly wrong degrades output rather than visibly breaking, which is exactly why the engine has to implement it correctly.

Once tokenized, inference runs in two phases.

-   **Prefill** processes the entire input sequence at once, computing attention for every input token and storing the results in the **KV cache** (Cached key and value tensors from the attention computation, kept so that previously-processed tokens do not have to be recomputed on every subsequent forward pass.). This determines time to first token.
-   **Decode** then runs one forward pass per output token, autoregressively, each pass attending over everything generated so far. This determines tokens per second.

Each decode pass has to end in a token, and neural networks emit vectors. The output layer produces one **logit** (An unnormalized score, one per vocabulary entry, produced by the model's output layer before softmax.) per vocabulary entry (tens or hundreds of thousands of them), which normalize into a probability distribution over every token the model knows.

> Figure 2.6. Turning logits into one token. The model emits one score per vocabulary entry; sampling decides which one becomes text. Temperature rescales the whole distribution, top-k truncates by rank, and top-p truncates by cumulative mass, which is why top-p adapts to the model's confidence and top-k cannot.Illustrative numbers

Setting temperature to 0, or top-k to 1, makes selection deterministic: the highest-probability token wins every time. Structured output pushes this further with logit biasing, masking tokens that would break a JSON schema before sampling ever happens. That machinery is what makes reliable tool calling possible, and implementing it correctly is squarely an inference concern.

Generation continues until the model emits a stop token, or until the context window or a `max_tokens` limit intervenes.

### 2.2.1 LLM architecture

Every model on Hugging Face ships a `config.json`: a few dozen lines fixing layer count, hidden dimensions, head counts, vocabulary size. Architecture is the set of decisions frozen at training time, and it determines what inference optimizations are even available to you.

Within one architecture there can be:

-   **Multiple sizes**, like an 8B and a 70B of the same family.
-   **Multiple variants**, like base and instruct, which share an architecture entirely.
-   **Unlimited fine-tunes.** LoRA and friends change behaviour without changing architecture, which is precisely why you can serve many of them from one base model.

### 2.2.2 Transformer blocks

A transformer is the same block repeated dozens of times. Each block holds an attention mechanism, a feed-forward network, normalization layers, and residual connections that let information skip past the block entirely.

The repetition is convenient for us. Optimize one block and you have optimized all eighty, which is why kernel-level work pays back so disproportionately here compared to typical application code.

### 2.2.3 Attention

Attention lets every token look at every other token and decide what is relevant. Three matrices (queries, keys, and values) are derived from the input, and the mechanism scores each query against every key, normalizes the scores, and uses them to take a weighted sum of values.

The important structural fact for inference is that scoring every position against every other is quadratic in sequence length. Double the context and attention’s intermediate matrices quadruple. That single property drives most of [section 2.5](https://learn-inference.com/chapters/models/optimizing-attention), a good part of [Chapter 5](https://learn-inference.com/chapters/techniques), and the existence of FlashAttention.

It is also why the KV cache exists. Without it, every decode step would recompute keys and values for the entire sequence so far, turning a linear generation into a quadratic one.

> Figure. Where the VRAM goes. Weights are fixed the moment you pick a model and a precision. The KV cache is not: it grows with context length and with every concurrent request, and it is what actually caps your batch size.Illustrative numbers

### 2.2.4 Mixture of Experts models

A Mixture of Experts model replaces the feed-forward network in each block with many parallel experts and a router that picks a few per token. A 235-billion-parameter MoE might activate just twenty-two billion parameters for any given token.

This is a genuinely strange object from a serving perspective. Total parameters set your memory requirement, because every expert must be resident somewhere. Active parameters set your compute and bandwidth per token. So for a single request an MoE is cheap to run and expensive to hold. Under batching, though, different requests light up different experts, and most of the model ends up active anyway.

It also creates a routing problem that does not exist for dense models: which GPU holds the expert this token needs, and what does it cost to get there. That question is the whole subject of expert parallelism in [Chapter 5](https://learn-inference.com/chapters/techniques/parallelism).
