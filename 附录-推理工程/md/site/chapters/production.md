# Production

<!-- https://learn-inference.com/chapters/production -->

A perfectly optimized runtime that cannot get a GPU when traffic arrives is a slow service. This chapter is everything around the model, which is where most production incidents actually live.

The techniques in the previous chapters make one replica fast. Nothing in them helps when you need forty replicas in ninety seconds, or when your region is out of H100s, or when a deploy takes the service down for eight minutes. Those are different problems with different answers.

## 6 sections

1.  [7.1ContainerizationDependency management · NIMs](https://learn-inference.com/chapters/production/containerization)
2.  [7.2AutoscalingConcurrency and batch sizing · Cold starts · Routing, load balancing, and queueing · Scale to zero · Independent component scaling](https://learn-inference.com/chapters/production/autoscaling)
3.  [7.3Multi-cloud capacity managementGPU procurement · Geo-aware load balancing · Building for reliability · Security and compliance](https://learn-inference.com/chapters/production/multi-cloud)
4.  [7.4Testing and deploymentZero-downtime deployment · Cost estimation · Observability](https://learn-inference.com/chapters/production/testing-deployment)
5.  [7.5Client codeClient latency overhead · Asynchronous inference · Streaming and protocol support](https://learn-inference.com/chapters/production/client-code)
6.  [7.6Where this leaves you](https://learn-inference.com/chapters/production/closing)
