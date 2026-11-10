# Speculative decoding · Techniques

<!-- https://learn-inference.com/chapters/techniques/speculative-decoding -->

Decode is stubbornly serial: one forward pass, one token. That is the constraint speculation attacks.

A cheap speculator proposes several draft tokens. The target model verifies all of them in a single forward pass (verification is parallel even though generation is not) and accepts the longest prefix that matches what it would have produced. Accept N drafts and you have produced N+1 tokens for the cost of roughly one target pass.

Crucially, output is identical to unspeculated decoding. This is not an approximation; rejected drafts are discarded.

> Figure 5.3. Speculative decoding, and where it stops paying. Drafts are cheap but not free, and one rejection discards everything after it. Push acceptance down or draft length up and the speedup goes below 1, at which point you are paying for compute to go slower.Illustrative numbers

The economics come down to three numbers: what a draft token costs, how often drafts are accepted, and how long the draft is. Long drafts look attractive and pay much less than expected, because a rejection discards everything after it: acceptance compounds geometrically.

One more caveat that the arithmetic above hides. Speculation spends compute to save time, which is a good trade when compute is idle. At high batch sizes it is not idle, and those draft passes compete with real requests. Speculation is a latency optimization for lightly-batched serving, and it can reduce total throughput.

### 5.2.1 Draft-target speculative decoding

The original form: a small model from the same family drafts for a large one, say a 1B drafting for a 70B. It works because the two share a tokenizer and training distribution, so the small model is right about easy tokens, which are most tokens.

The cost is a second model to deploy, version, and keep resident in VRAM.

### 5.2.2 Medusa

Medusa removes the second model. It bolts extra decoding heads onto the target itself, each predicting a token further ahead, so drafts come from the model that is already loaded.

No separate model to deploy or hold resident, in exchange for limits on how many drafts the heads can produce and how often they are accepted, and a training step to fit the heads.

### 5.2.3 EAGLE

EAGLE is the current state of the art and the one most production deployments reach for. Rather than predicting tokens from the output distribution, it takes the target model’s hidden states as input and predicts at the feature level.

That extra context makes the drafts substantially more accurate, and since acceptance rate is what the entire technique turns on, better drafts translate almost directly into speedup. EAGLE-2 adds dynamic draft trees, letting the speculator explore several candidate continuations rather than committing to one line, and EAGLE-3 scales the approach further.

### 5.2.4 N-gram speculation and lookahead decoding

The cheapest speculators involve no model at all. N-gram speculation keeps a dictionary mapping recent prefixes to likely suffixes and drafts from that.

It sounds too crude to work, and for open-ended chat it mostly is. For workloads with heavy verbatim repetition it is remarkable: code editing where most of a file is reproduced unchanged, summarization that quotes the source, structured output with fixed schemas. The draft costs a string lookup rather than a forward pass, and acceptance is high on exactly the workloads where a neural drafter would be overkill.
