# Measuring latency and throughput · Prerequisites

<!-- https://learn-inference.com/chapters/prerequisites/latency-throughput -->

Two numbers dominate LLM serving. **TTFT** (Time to first token: how long the user waits before any output appears.) is how long before the first token appears, and it is governed by prefill, which is compute bound. **TPS** (Tokens per second: the rate at which output arrives after the first token.) is how fast tokens arrive afterwards, governed by decode, which is memory bandwidth bound. Different phases, different bottlenecks, different fixes.

> Figure 1.4. Time to first token versus tokens per second. Two numbers, two phases, two bottlenecks. A response that starts instantly and trickles can feel faster than one that pauses and then dumps, even when the second finishes first.Illustrative numbers

TTFT is unambiguous. TPS is not, and the ambiguity causes real confusion in benchmark comparisons, because the same three letters describe both a latency metric and a throughput metric. When precision matters, say which you mean:

-   **Perceived TPS** is tokens per second for one user. A latency metric.
-   **Total TPS** is tokens per second across the whole service. A throughput metric, and typically far larger.
-   **Inter-token latency** is the gap between consecutive tokens. 10 ms of ITL is 100 tokens per second per user.

Both metrics assume streaming output and a human reading it. When the output is a tool call an agent will parse, individual tokens have no value on their own and the only number that means anything is total response time.

### 1.4.1 Latency percentiles

Mean latency is the most commonly reported figure and a poor summary of the distribution. Response times are right-skewed: most cluster near a mode, and a long tail stretches out to the right. The mean gets dragged toward the tail without describing it, so it simultaneously overstates the typical experience and understates the bad one.

> Figure 1.5. Why the mean hides your worst requests. Response times cluster and then trail off to the right. Push the tail weight up and watch the mean barely move while P99 runs away. That gap is the one in a hundred users who thinks your product is broken.Constants from the book

Percentiles say something checkable instead. P90 means one request in ten is slower than this; P99 means one in a hundred. Those outliers do disproportionate damage, because a product where nine interactions feel instant and the tenth hangs for several seconds gets remembered as unreliable rather than fast.

Pushing the mean down is worth doing. Pushing P90 and P99 down is what makes a product feel dependable, and it is usually the harder and more valuable work.

### 1.4.2 End-to-end metrics

The last distinction is where you start and stop the clock. Inference time is on-GPU time: how long generation itself took. End-to-end time is what the user experienced, including network round trips, queueing, load balancer hops, and whatever your client code does before and after.

Keep both. Inference time tells you whether your model performance work is landing. End-to-end time tells you whether any of it reached the user.

The gap between them is diagnostic. When inference time is excellent and end-to-end time is poor, more kernel work will not help you: the problem is in the infrastructure, and [Chapter 7](https://learn-inference.com/chapters/production) is where to look.
