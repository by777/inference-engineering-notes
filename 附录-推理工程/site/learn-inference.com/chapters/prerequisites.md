# Prerequisites

<!-- https://learn-inference.com/chapters/prerequisites -->

Optimization means picking the best option from a set. Until your product defines what “best” means, performance work has no criterion to optimize against, and a benchmark cannot supply one.

Almost nothing in inference is a pure win. You buy throughput with latency, latency with money, and money with quality. Which of those trades is acceptable depends entirely on what you are building, and that makes the product questions genuine prerequisites rather than preamble.

The useful frame is specialization. Football players are big, fast, and strong, but they are not as big as sumo wrestlers, as fast as sprinters, or as strong as powerlifters. They are built for the specific demands of a position across a season. An inference service is the same: the more constraints you can name, the better it can be made. A system that must serve every workload well serves none of them excellently.

Before optimizing anything, you should be able to answer five questions.

-   **Which model?** Everything downstream depends on this one, and it is the decision with the largest effect on both cost and speed.
-   **What is the interface?** How inputs arrive and what shape the output has to take.
-   **What is the latency budget?** End to end, from the user’s action to something appearing on their screen.
-   **What are the unit economics?** What a request, a user, or a month can reasonably cost.
-   **What do usage patterns look like?** Concurrency, and whether it follows business hours, a launch, or nothing at all.

Early on, most of these have no honest answer yet. That is a fine reason to use an off-the-shelf API and leave inference engineering alone until the questions have answers. The work is worth doing once the constraints are real.

## 4 sections

1.  [1.1Scale and specialization](https://learn-inference.com/chapters/prerequisites/scale-and-specialization)
2.  [1.2About your appAI-native applications · Online versus offline · Consumer versus B2B](https://learn-inference.com/chapters/prerequisites/about-your-app)
3.  [1.3Model selectionModel evaluation · Fine-tuning for domain-specific quality · Distillation](https://learn-inference.com/chapters/prerequisites/model-selection)
4.  [1.4Measuring latency and throughputLatency percentiles · End-to-end metrics](https://learn-inference.com/chapters/prerequisites/latency-throughput)
