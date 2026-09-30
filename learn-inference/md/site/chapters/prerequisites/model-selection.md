# Model selection · Prerequisites

<!-- https://learn-inference.com/chapters/prerequisites/model-selection -->

Hold hardware, runtime, and optimizations constant, and a smaller model is faster and cheaper than a larger one. Nothing else you do moves the numbers as much.

Which makes model choice, not engine choice or speculation algorithm, the highest-leverage decision in inference performance. It is also the one most often treated as settled before the performance conversation starts.

Before product-market fit, use a frontier model through an API and do not think about this. After it, invert the advice completely: find or build the smallest model that still passes your evaluations. Sometimes that is still a trillion-parameter frontier model. It is always worth checking whether it is not.

Architecture choice has a second, less obvious effect. Inference engines support architectures unevenly, and an unusual model can quietly cut you off from half the optimization toolkit. Popular architectures come with better support across every engine, which is worth real performance.

### 1.3.1 Model evaluation

Evaluations measure whether a model is good at your task. They are a prerequisite for inference work for two reasons: they stop you spending a month making a useless model fast, and they give you the baseline you need before applying any technique that might cost quality.

That second reason is the one people skip. Quantization is the big one (the lossy member of [Chapter 5](https://learn-inference.com/chapters/techniques)'s toolkit), and without a baseline you will not notice the degradation until a user does.

Public benchmarks like MMLU or SWE-bench are useful for drawing up a shortlist and useless for anything after that. They saturate, and labs have every incentive to optimize against them directly. Goodhart’s law applies with unusual force here: when a measure becomes a target, it stops being a good measure.

Three things make evaluation work actually pay off. Look at your own data rather than only at aggregate scores. Be precise about the hardest cases your model must handle and concentrate there. And use existing tooling, because this is a well-trodden problem and rebuilding it is not where your advantage lies.

### 1.3.2 Fine-tuning for domain-specific quality

Fine-tuning adapts a pre-trained model to a narrower task by training it further on new data. The architecture stays the same; the weights move.

For inference this is a lever on size. If a small fine-tuned model passes evaluations that only a large general model passed before, every latency and cost target afterwards becomes easier.

Text-to-SQL is the standard example, and it is close to the best case. General coding models write good SQL, but they are hundreds of billions of parameters and know a hundred other languages you are not using. SQL is small and highly structured, so a few-billion parameter model fine-tuned on it can match them on that one task.

Most domains will not compress that far. What the example shows is what becomes possible when the domain is genuinely narrow, the evaluation criteria are sharp, and you have clean labelled data. Miss any of those three and fine-tuning tends to disappoint.

### 1.3.3 Distillation

Distillation trains a small student model to imitate a large teacher. It differs from fine-tuning on synthetic data in what the student sees: not just the teacher’s final answers, but its full probability distributions over tokens. That is a much richer signal, and it transfers behaviour rather than facts.

The distinction is worth holding onto. Fine-tuning makes a model better in a domain. Distillation makes a model behave like another model, including the parts of that behaviour you would rather it did not copy.

In practice distillation is far less common than fine-tuning. When a lab ships a family of sizes, the small members are usually trained independently rather than distilled, specifically so the large model’s biases do not put a ceiling on them. Where it earns its keep is when a lab has trained only a frontier-scale model and wants it to reach hardware it otherwise never would.

DeepSeek-R1 is the well-known case. At 671 billion parameters the model itself was out of reach for most people, so alongside it DeepSeek released versions distilled onto Llama 3 and Qwen 2.5 architectures. Those distills scored lower on benchmarks but showed similar reasoning behaviour, and because they sat on popular architectures they inherited all the existing performance work for free. That last part is the underrated half of the decision.
