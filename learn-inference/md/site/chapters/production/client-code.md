# Client code · Production

<!-- https://learn-inference.com/chapters/production/client-code -->

The last few hundred milliseconds are on the client, and they are routinely ignored by teams who have spent months on the server.

### 7.5.1 Client latency overhead

TLS handshakes, DNS lookups, connection setup, and serialization all happen before your model sees anything. Re-use connections, keep them alive, and put a CDN or edge terminator close to users so the handshake happens near them rather than near your GPUs.

None of this is interesting work. It is regularly worth more milliseconds than the kernel optimization it gets skipped in favour of.

### 7.5.2 Asynchronous inference

Not every request needs an open connection. Batch jobs, long video generation, and bulk transcription are better served by submitting work, getting an identifier, and collecting the result later via polling or a webhook.

This decouples client timeouts from server duration entirely, which removes a whole category of failure. It also lets the scheduler defer that work to fill capacity that would otherwise idle.

### 7.5.3 Streaming and protocol support

Server-sent events are the default for token streaming: unidirectional, simple, and well-supported everywhere. WebSockets add bidirectionality, which is what live transcription and voice agents need: audio flowing up while text or audio flows down. gRPC gives bidirectional streaming with structured schemas, at the cost of more client-side machinery.

Streaming is the highest-leverage perceived-latency change available, and it happens entirely at the API layer. Nothing about the model changes; the user starts seeing output at your TTFT instead of your total response time. For a long response that is the difference between a product that feels fast and one that feels broken, which is where this book started, in [Chapter 1](https://learn-inference.com/chapters/prerequisites/latency-throughput).
