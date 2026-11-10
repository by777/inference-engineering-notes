# Modalities

<!-- https://learn-inference.com/chapters/modalities -->

Most non-text models are transformer variants, which means the toolkit transfers almost intact. Image and video generation are the exception, with a different architecture and different bottlenecks.

Vision-language models, multimodal embeddings, speech recognition, and speech synthesis all adapt LLM architectures. That is good news: the same engines serve them, the same quantization and batching and caching apply, and the intuitions from [Chapter 2](https://learn-inference.com/chapters/models) hold.

What changes is the metric. Time to first token is meaningless for a model whose output is an image, and the first 20 milliseconds of audio from a TTS model is not useful on its own. Each modality measures something different, and knowing which number matters is most of the work.

## 6 sections

1.  [6.1Vision language modelsVideo processing for vision language models · Omni-modal models](https://learn-inference.com/chapters/modalities/vlms)
2.  [6.2Embedding modelsEmbedding model architecture · Embedding model inference](https://learn-inference.com/chapters/modalities/embeddings)
3.  [6.3ASR modelsSingle-chunk latency optimization · Long file latency optimization · Diarization](https://learn-inference.com/chapters/modalities/asr)
4.  [6.4TTS modelsStreaming real-time text to speech · Speech-to-speech models](https://learn-inference.com/chapters/modalities/tts)
5.  [6.5Image generation modelsImage generation kernel optimization · One weird trick for faster image generation](https://learn-inference.com/chapters/modalities/image-models)
6.  [6.6Video generation modelsAttention optimization and quantization · Context parallelism](https://learn-inference.com/chapters/modalities/video-models)
