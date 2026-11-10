# TTS models · Modalities

<!-- https://learn-inference.com/chapters/modalities/tts -->

Text to speech inverts ASR. Modern TTS is usually a fine-tuned LLM emitting audio tokens, which means most of the LLM toolkit applies: quantization, in-flight batching, the same optimized engines.

### 6.4.1 Streaming real-time text to speech

The metric is time to first audio chunk. Nobody needs the whole utterance before playback starts; they need enough to begin speaking, and the rest can arrive while the first clause plays.

Which means a well-built TTS stage can start producing audio from a partial LLM response. You do not wait for the sentence to finish; you start speaking the first clause while the model is still writing the second. That overlap is the difference between a voice agent that feels responsive and one that does not.

### 6.4.2 Speech-to-speech models

Speech-to-speech models collapse the three-stage pipeline into one, taking audio in and producing audio out without a text intermediate. Removing two stages removes two sets of latency and two failure modes, and preserves tone and emotion that transcription discards.

The costs are real: today’s speech-to-speech models are less capable and more expensive than the cascade, and you lose the text transcript most products need for logging and moderation. Worth it for some products, not for most.

Why voice is unforgiving

The budget covers three models plus the network, in series, inside a conversational turn. And voice is where users notice immediately, because everyone has a lifetime of calibration for how long a person takes to answer.

> Figure. The voice agent latency budget. Three models in series plus network, and the whole chain has to fit inside a conversational turn. No single stage is slow. The sum is the problem, which is why a voice pipeline is budgeted end to end rather than model by model. The marks are rules of thumb for when the pause becomes audible, not hard limits.Illustrative numbers
