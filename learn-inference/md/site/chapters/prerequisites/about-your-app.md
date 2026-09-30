# About your app · Prerequisites

<!-- https://learn-inference.com/chapters/prerequisites/about-your-app -->

Every decision that follows is downstream of your use case. A basketball coach and a gymnastics coach walk into the same school and recruit opposite children; neither is wrong.

Two situations require building for full generality: you have trained a foundation model and are selling access to it, or you are building an inference platform that has to serve models you have not seen yet. Everything else is a vertical application, and vertical applications should be greedy about constraints.

### 1.2.1 AI-native applications

The categories below are a small sample, but they show how far apart the requirements sit. Each one is trying to make a different number small.

> Figure. What each kind of product optimizes for. Describe what you are building with the two controls; the figure answers with the number to make small. The archetypes light up as they match. Search lights in more than one place because it is two workloads sharing one name, and three products share the person-and-one-call shape, separated only by what failure means.

### 1.2.2 Online versus offline

The central trade in inference is **latency** (How long one request takes, measured from the user's perspective.) against **throughput** (How much total work the service completes per unit time, across all users.). Lower latency makes the product feel faster. Higher throughput makes it cheaper, because the same traffic fits on fewer GPUs.

Most applications are online. Somebody is waiting, so latency wins. But a surprising amount of real inference work has nobody waiting at all: transcribing a back catalogue of podcasts, re-embedding a document corpus on a schedule, cleaning data for a training run. These are batch jobs, and for batch jobs an individual request being slow is irrelevant as long as the system chews through the queue.

The same model often serves both. Whisper might power a live dictation feature and a bulk transcription pipeline. If both have real volume, run two deployments of the same weights with opposite configurations rather than one compromise that is mediocre at each.

### 1.2.3 Consumer versus B2B

Consumer products are cost-sensitive and their traffic is unpredictable. Many are built for virality, which means a single post can multiply your load overnight. Optimize for marginal cost and for the ability to scale sideways in a hurry; keep latency and availability respectable rather than exceptional.

B2B products usually have better margins and steadier load, and in exchange they are held to a much higher standard. Software sitting in someone else’s revenue path does not get to be slow or occasionally unavailable. Favour latency and uptime; treat cost as the important secondary concern it is.

Compliance narrows the options

In regulated industries this stops being an afterthought and starts constraining which regions you may put GPUs in, what may be logged, and which providers you may use at all. Data sovereignty, user privacy, and your providers’ own compliance posture all limit the search space before performance work begins. Get security and legal involved early rather than rebuilding later.
