# Where to put the abstraction · Inference

<!-- https://learn-inference.com/chapters/inference/abstraction -->

Once the runtime and infrastructure work exists, somebody has to decide how it is exposed. This applies whether you are buying inference from a provider or building a platform for your own engineers; the question is the same either way.

At one extreme, inference is a black box: hand over weights, receive an API. At the other, you get compute, network, and disk, and everything above that is yours. Both extremes are defensible and most teams belong somewhere between them.

> Figure. How much of the stack do you want to own?. Control and productivity trade against each other along one axis. Each step down buys the ability to tune deeper layers and costs the engineering time to run them. Take as much abstraction as your requirements allow, and give up productivity only where something you need demands the control.

I would push most teams further toward abstraction than instinct suggests. Owning the whole stack is satisfying and it is occasionally correct, but it is a standing commitment of engineering time that competes directly with the product. Keep only as much control as you have a concrete use for.
