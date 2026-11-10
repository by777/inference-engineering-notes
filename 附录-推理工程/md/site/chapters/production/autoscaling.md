# Autoscaling · Production

<!-- https://learn-inference.com/chapters/production/autoscaling -->

Autoscaling adjusts how many replicas a model has. There are two signals to drive it from, and they disagree more often than you would expect.

-   **Utilization**: GPU memory and compute usage. It is a lagging indicator: by the time utilization is high, the queue has already formed.
-   **Traffic**: requests in the system. Available earlier, so it can be acted on proactively.

They come apart under uneven load. A handful of requests with hundreds of thousands of uncached input tokens will pin utilization while traffic counts look calm; a flood of small requests with high cache hit rates will do the opposite. Use both.

Five knobs shape a traffic-based autoscaler:

-   **Min replicas**, the floor that stays running regardless.
-   **Max replicas**, the ceiling when traffic is high.
-   **Autoscaling window**, the sliding period over which traffic is measured.
-   **Scale-down delay**, how long to wait before releasing capacity in case the spike returns.
-   **Concurrency target**, how many requests one replica should handle at once.

> Figure 7.4. Autoscaling across a spiky day. The gap between the two lines is queued requests. Cold start time decides how wide it gets: push it up and no scaling policy can keep the fleet ahead of a spike, which is why cold starts get their own section.Illustrative numbers

Every one of those settings is a bet. A longer scale-down delay protects you from flapping on spiky traffic and costs money when traffic genuinely cools. There is no correct value, only a value matched to your traffic shape and your tolerance for each failure.

### 7.2.1 Concurrency and batch sizing

The concurrency target is where autoscaling meets [batching](https://learn-inference.com/chapters/models/bottlenecks), and it is the setting people most often get backwards.

Set it too low and every replica runs at batch sizes that waste the GPU: you are back at the left side of the roofline, paying for tensor cores that idle while weights stream from memory. Set it too high and requests queue inside a replica, which is invisible to an autoscaler watching replica count and plainly visible to users watching p99.

Find it empirically. Sweep concurrency against latency for your actual workload and pick the point just before latency starts climbing steeply. That knee is your target.

### 7.2.2 Cold starts

A cold start is the time to bring up a new replica, and it quietly governs the whole autoscaling system. If you cannot add capacity quickly, you cannot scale down confidently, so you over-provision permanently and pay for it every hour of every day.

> Figure 7.9. What a cold start is actually made of. Four stages, each attacked separately (the durations here are representative, not measured). The mitigations are not equally valuable: a warm pool removes the one stage you do not otherwise control, because procurement speed is mostly a function of your cloud provider.Illustrative numbers

Attack the four stages separately, because they respond to completely different fixes. Procurement is solved with a warm pool of nodes you flex between models. Image loading is solved by making the image smaller and by giving the node more bandwidth to pull it. Weight loading is solved by streaming from a local cache rather than pulling from object storage. Engine startup is solved by caching the compiled artifact, which is the difference between seconds and minutes on TensorRT-LLM.

### 7.2.3 Routing, load balancing, and queueing

Round-robin is the wrong default for inference. Requests are wildly uneven (a 200-token request and a 200,000-token request are not interchangeable), so balancing on connection count sends work to replicas that are already deeply busy.

Better signals are queue depth and, where it applies, [cache locality](https://learn-inference.com/chapters/techniques/caching). Routing a request to the replica that already holds its prefix can matter more than routing it to the least busy one, which is a unusual property of this workload.

Queue somewhere. Without a queue, a spike becomes dropped connections; with one, it becomes latency you can observe and reason about. Bound it, and shed load deliberately when the bound is hit: a fast rejection is a better product experience than a request that hangs for ninety seconds and then fails.

### 7.2.4 Scale to zero

Scaling to zero means paying nothing when nobody is using a model, and paying a full cold start for whoever arrives first.

Correct for development environments, internal tools, and the long tail of models that see a handful of requests a day. Wrong for anything user-facing where that first request is somebody’s first impression. The economics only work when your cold start is fast.

### 7.2.5 Independent component scaling

Multi-model pipelines (the voice stack from [Chapter 6](https://learn-inference.com/chapters/modalities/tts), or a retrieval system with an embedder and a reranker and an LLM) should not scale as one unit.

Each component has its own throughput and its own hardware fit. Whisper is happy on a fractional GPU; the LLM needs a whole one. Bundling them means scaling all three because one is busy, which wastes the other two. Deploy separately and let each follow its own demand. This is [disaggregation](https://learn-inference.com/chapters/techniques/disaggregation) applied to a pipeline rather than to phases of one model.
