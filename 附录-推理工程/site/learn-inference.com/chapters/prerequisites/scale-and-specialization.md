# Scale and specialization · Prerequisites

<!-- https://learn-inference.com/chapters/prerequisites/scale-and-specialization -->

There are two ways to get a model into your product.

-   **Shared inference:** send traffic to a public endpoint and pay per million tokens.
-   **Dedicated deployment:** rent GPUs, run the service yourself, and pay per hour whether or not anyone is using it.

This is a related but separate axis from open versus closed weights. Plenty of open models are available on shared endpoints, and large customers can often get dedicated capacity for closed models. The connection is that open weights are what make unrestricted dedicated inference possible at all.

Almost every product should start shared. Zero overhead, no cold starts, no capacity planning, and you only pay for what you use. What you give up is control: your uptime is capped by theirs, your latency is whatever they decide, and your cost scales linearly forever.

That last point is the one that eventually forces the move. Linear cost is a bargain at low volume and a problem at high volume, and the crossover comes later than a raw per-token comparison suggests, because a dedicated GPU bills for the hours you are asleep too.

> Figure. When per-token billing stops being the cheap option. Shared endpoints cost nothing at zero and scale linearly. Dedicated GPUs are a step function: flat until you saturate a card, then another card. Drop utilization to what your traffic really looks like overnight and the crossover moves a long way right.Illustrative numbers

Utilization is the term to keep in view. A GPU that is busy 20 percent of the day costs the same as one busy 100 percent of the day, so a napkin calculation using peak throughput will tell you to migrate before you should. Three things actually justify the switch:

-   **Scale:** your volume makes per-GPU cheaper than per-token, honestly accounting for idle time.
-   **Specialization:** you are running a fine-tuned model, or you have latency or uptime requirements a shared endpoint will not meet.
-   **Orchestration:** your product chains several models together and the network hops between them have become the latency.

Moving makes you responsible for your own inference engineering. That is more control and more surface area, and it raises the floor on your monthly spend from zero to the cost of the smallest sensible fleet. Do it when there is a clear business reason, not because it feels more serious.
