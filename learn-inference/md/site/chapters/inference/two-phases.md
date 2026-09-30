# Two phases, two disciplines · Inference

<!-- https://learn-inference.com/chapters/inference/two-phases -->

Every generative model passes through the same two phases. **Training** (Learning model weights from data. Happens once per model version, with a budget and a known end date.) is where weights are learned from data. **Inference** (Running a trained model to produce output for a user request. Happens continuously, at whatever rate your users demand.) is where those weights are put to work in production. They look adjacent on a diagram and they are almost nothing alike in practice.

Training is a project. It has a start, a budget, a cluster reserved for a known number of weeks, and a moment where it is finished. Inference is an operation. It has no end date, its load is set by other people, and its cost scales with your success. The failure modes differ too: a bad training run wastes its budget, while a bad inference service means the product itself is down.

Anyone who worked through the previous decade of machine learning already knows a version of this lifecycle. That version was easy. Serving a gradient-boosted tree meant a CPU, a Flask app, and a model file measured in megabytes. You could hold the whole thing in your head, and the interesting problems were all upstream in feature engineering.

Generative models broke that. The weights are large enough that where they sit in memory is an architectural decision. Output is produced one token at a time, so a single response is a few hundred sequential passes through the network rather than one. And demand is spiky in a way that batch scoring never was. You cannot take weights, rent a GPU, and expect the result to be fast or reliable enough to put in front of users. Something has to be built around it.

What this book means by inference

Not the mathematical sense of the word, and not the training-time forward pass. Inference here means the whole production concern: a model, the software serving it, the hardware underneath, and the systems that keep all of it available when traffic arrives.
