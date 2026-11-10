# ASR models · Modalities

<!-- https://learn-inference.com/chapters/modalities/asr -->

Automatic speech recognition takes audio and produces text. Whisper is the dominant open model, and the largest variant is 1.55B parameters: small enough to transcribe faster than real time on a fraction of a modern GPU.

Whisper is encoder-decoder: the encoder processes a 30-second audio window, the decoder autoregressively emits text. That 30-second limit is a hard architectural constraint and it shapes both of the optimization problems below.

### 6.3.1 Single-chunk latency optimization

For live transcription, the number to watch is round-trip time for one chunk. Roughly 200 milliseconds is the target, because that is about human reaction time.

On an optimized engine there is little left to win at the runtime layer: the model is small and already fast. The gains are in orchestration, and the big one is streaming, implemented at the API server rather than the model.

A WebSocket carries audio in and text out continuously. A voice activity detection model watches the stream and segments it into chunks, and each chunk runs through ASR normally. Keeping consecutive chunks on the same GPU has a second benefit: the previous chunk’s output can prefix the next one, which improves transcription quality.

### 6.3.2 Long file latency optimization

An hour-long podcast is 120 thirty-second chunks, and the useful property is that they are independent. Transcription of a long file is an embarrassingly parallel problem, so the latency floor is set by how wide you are willing to fan out rather than by the length of the file.

This is the throughput-versus-latency trade from [Chapter 1](https://learn-inference.com/chapters/prerequisites/about-your-app) in its clearest form. The seams need care: a chunk boundary through the middle of a word produces a transcription error, so a VAD model cuts at silence rather than at fixed intervals, and the transcripts are stitched back together by timestamp.

### 6.3.3 Diarization

Diarization answers “who spoke when”, and it is a separate model from transcription. Running it means a second pipeline whose output has to be aligned with the first.

It is also harder to parallelize, because speaker identity is a property of the whole recording. A speaker in chunk 90 has to be recognized as the same person from chunk 3, which resists the chunk-independence that makes transcription easy to scale.
