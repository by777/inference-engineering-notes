# Multi-cloud capacity management · Production

<!-- https://learn-inference.com/chapters/production/multi-cloud -->

Past a few hundred GPUs the problem stops being “how do I scale” and becomes “where are there GPUs”. That is a different discipline, and it is where inference stops being a software problem.

### 7.3.1 GPU procurement

Capacity is contended and geographically lumpy. On-demand pricing is highest and most flexible; reserved capacity is cheaper and commits you; spot is cheapest and can be reclaimed on minutes of notice.

Most serious deployments blend all three: reserved for baseline, on-demand for normal variation, spot for batch work that tolerates interruption. And they hold relationships with several providers, because the answer to “can I have forty more H100s today” varies by vendor and by week.

### 7.3.2 Geo-aware load balancing

Network latency is physics. A round trip from Sydney to us-east-1 costs more than 200 milliseconds before your model does anything, which for a [voice agent](https://learn-inference.com/chapters/modalities/tts) has already spent the entire budget.

Route users to their nearest region with capacity, and accept that those two conditions conflict. Falling back to a distant region beats queueing locally: a slower answer is better than no answer, and this is exactly the tradeoff a geo-aware balancer exists to make.

### 7.3.3 Building for reliability

Closed-model APIs sit around two nines. Dedicated deployments can reach four, and the difference is almost entirely architectural rather than about any single component being better.

Spread across regions and providers so no single failure takes everything. Health-check on real inference rather than on a process being alive: a replica in a bad CUDA state will answer your `/health` endpoint cheerfully while failing every actual request. Fail over automatically, and rehearse it.

### 7.3.4 Security and compliance

Where GPUs physically sit becomes a legal question in regulated industries. Data sovereignty may forbid a region outright, regardless of capacity or latency, and that constraint is not negotiable by engineering.

Beyond placement: encrypt in transit and at rest, be deliberate about what inference logs retain (prompts routinely contain exactly the data you promised not to store), and confirm your providers hold the certifications your customers will ask about. Involve security and legal before you design around a region, not after.
