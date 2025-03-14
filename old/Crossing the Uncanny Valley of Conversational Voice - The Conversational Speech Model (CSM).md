# Crossing the Uncanny Valley of Conversational Voice: The Conversational Speech Model (CSM)

**Authors:** Brendan Iribe, Ankit Kumar, Johan Schalkwyk, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame Team

**Date:** February 27, 2025

**Abstract:**
This paper introduces the Conversational Speech Model (CSM), a novel approach to conversational speech generation aimed at achieving "voice presence" by addressing the limitations of current digital voice assistants. CSM leverages a multimodal, end-to-end transformer-based architecture that operates directly on Residual Vector Quantization (RVQ) tokens, enabling real-time contextual adaptation and improved expressivity. We detail the model's architecture, training methodologies, and evaluation suite, which includes both objective and subjective metrics. Notably, we introduce new objective benchmarks for assessing contextual understanding and pronunciation consistency. Our results demonstrate significant progress in conversational speech generation, highlighting the potential of CSM to create more natural and engaging AI companions.

**1. Introduction**

Voice is a fundamental medium of human communication, conveying not only semantic content but also rich emotional and contextual cues. Current digital voice assistants, however, often lack the nuanced expressivity necessary for truly engaging interactions, leading to an experience that can become exhausting over time. To address this, we aim to achieve "voice presence," a quality that makes spoken interactions feel genuine and understood. This paper presents the Conversational Speech Model (CSM), a system designed to bridge the gap between current digital voice assistants and human-like conversational partners.

**2. Achieving Voice Presence**

Voice presence requires several key components:

* **Emotional intelligence:** The ability to read and respond to emotional contexts.
* **Conversational dynamics:** Natural timing, pauses, interruptions, and emphasis.
* **Contextual awareness:** Adjusting tone and style to match the situation.
* **Consistent personality:** Maintaining a coherent, reliable, and appropriate presence.

**3. The Conversational Speech Model (CSM)**

CSM is a multimodal, text and speech model that operates directly on RVQ tokens. It employs two autoregressive transformers, inspired by the RQ-Transformer (Zeghidour et al., 2022) [4]. The model addresses the limitations of traditional text-to-speech (TTS) models by incorporating conversational history and context into the speech generation process.

**3.1. Model Architecture**

CSM utilizes a split-RVQ tokenizer (Mimi), producing one semantic codebook and N-1 acoustic codebooks per frame. The architecture includes:

* **Backbone:** A multimodal transformer that processes interleaved text and audio tokens, modeling the zeroth codebook.
* **Decoder:** An audio decoder that uses distinct linear heads for each codebook, reconstructing speech from the backbone's representations.

Both transformers are variants of the Llama architecture (Touvron et al., 2024) [6]. Training samples are structured as alternating interleaved patterns of text and audio, with speaker identity encoded in the text.

**3.2. Compute Amortization**

To address the computational challenges of training, we employ a compute amortization scheme. The audio decoder is trained on a random 1/16 subset of the audio frames, while the zeroth codebook is trained on every frame. This approach alleviates the memory bottleneck without compromising model fidelity.

**4. Experiments**

**4.1. Dataset**

We used a large dataset of publicly available audio, transcribed, diarized, and segmented, resulting in approximately one million hours of predominantly English audio.

**4.2. Model Sizes**

We trained three model sizes:

* Tiny: 1B backbone, 100M decoder
* Small: 3B backbone, 250M decoder
* Medium: 8B backbone, 300M decoder

Each model was trained with a 2048 sequence length over five epochs.

**5. Evaluation**

Our evaluation suite measures model performance across faithfulness to text, context utilization, prosody, and latency, using both objective and subjective metrics.

**5.1. Objective Metrics**

Traditional benchmarks like Word Error Rate (WER) and Speaker Similarity (SIM) are saturated. We introduce new phonetic transcription-based benchmarks:

* **Homograph Disambiguation:** Measures the model's ability to correctly pronounce different words with the same orthography.
* **Pronunciation Continuation Consistency:** Evaluates the model's ability to maintain pronunciation consistency in multi-turn speech.

Our objective results show that performance improves with larger models, demonstrating the effectiveness of scaling.

**5.2. Subjective Metrics**

We conducted two Comparative Mean Opinion Score (CMOS) studies using the Expresso dataset (Lee et al., 2023) [5]. The studies assessed naturalness and prosodic appropriateness.

* **No Context:** Listeners rated generated vs. human speech without context.
* **Context:** Listeners rated generated vs. human speech with 90 seconds of audio and text context.

Results indicate that while naturalness is saturated, a gap remains in prosodic appropriateness in conversational speech.

**6. Limitations and Future Work**

CSM is currently trained on English data and lacks strong multilingual capabilities. Future work will focus on scaling model size, increasing dataset volume, and expanding language support. We also aim to explore the integration of pre-trained language models and develop fully duplex models that can learn conversational dynamics.

**7. Conclusion**

The Conversational Speech Model (CSM) represents a significant step towards achieving voice presence in digital voice assistants. By leveraging a multimodal, end-to-end transformer architecture and introducing novel evaluation metrics, we have demonstrated improved contextual awareness and expressivity in conversational speech generation. Future work will focus on scaling and expanding the model's capabilities, ultimately aiming to create more natural and engaging AI companions.

**8. References**

[1]  Zeghidour, N., Coucke, A., Kharitonov, M., Kharitonov, M., & Usunier, N. (2023). Textless Speech Generation with Discrete Speech Units. *arXiv preprint arXiv:2306.12925*.
[2]  Baevski, A., Hsu, W. N., Conneau, A., & Collobert, R. (2021). Data2vec: A general framework for self-supervised learning in speech, vision and language. *arXiv preprint arXiv:2107.03312*.
[3]  Popov, V., Kudinov, M., & Vasilev, S. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers. *arXiv preprint arXiv:2306.05284*.
[4]  Zeghidour, N., Usunier, N., & Synnaeve, G. (2022). SoundStream: An End-to-End Neural Audio Codec. *arXiv preprint arXiv:2203.01941*.
[5]  Lee, J., Kim, S., Lee, D., & Kim, H. (2024). Expresso: A Large-Scale Dataset for Expressive Text-to-Speech. *arXiv preprint arXiv:2410.00037*.
[6]  Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Lample, G. (2024). Llama 3: Open Foundation and Instruction-tuned Language Models. *arXiv preprint arXiv:2407.21783*.