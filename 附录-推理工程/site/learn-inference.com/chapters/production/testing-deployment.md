# Testing and deployment · Production

<!-- https://learn-inference.com/chapters/production/testing-deployment -->

### 7.4.1 Zero-downtime deployment

Deploy strategies written for stateless web services assume replicas start in seconds. Inference replicas take minutes, and any strategy that ignores that trades capacity for downtime.

Bring the new version up, send it a small slice of live traffic, verify it serves real requests, and ramp while the old fleet scales down underneath it. Running two full fleets side by side buys the same safety at twice the GPU bill, which is why canaries won this argument. And watch outputs, not just error rates: quality regressions from a new model or a new quantization show up in answers.

### 7.4.2 Cost estimation

Cost per token is the number the business cares about, and it falls out of three things you already know: GPU hourly rate, achieved throughput, and utilization.

Utilization is the term people omit, and it is the one that dominates. A GPU busy 30% of the day costs three times per token what the same GPU busy 90% of the day does, with identical hardware and identical optimization. Most inference cost problems are utilization problems rather than performance problems, which is the same point the [crossover calculator](https://learn-inference.com/chapters/prerequisites/scale-and-specialization) in Chapter 1 makes.

### 7.4.3 Observability

Instrument the numbers from [Chapter 1](https://learn-inference.com/chapters/prerequisites/latency-throughput) at percentiles, not averages: TTFT, inter-token latency, total response time, queue depth, batch size, cache hit rate, GPU utilization, and cost per token.

Two things separate useful inference observability from generic service monitoring. Measure end-to-end and on-GPU separately: the gap between them tells you whether to look at the model or the infrastructure. And monitor output quality, not just latency: a model that degrades after a config change produces perfectly fast, perfectly available, subtly worse answers, and nothing in a standard dashboard will tell you.

The alert that matters

Queue depth. It rises before latency does, which makes it the earliest honest signal that capacity is short. By the time p99 moves, users have already had the bad experience.
