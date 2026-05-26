# MiniCPM-V & o Cookbook

Cook up amazing AI applications effortlessly with **MiniCPM-V**, **MiniCPM-o**, and the **MiniCPM LLM** series — bringing text, vision, speech, and live-streaming capabilities right to your fingertips.

## What's new

- 🎉 **MiniCPM-V 4.6** released — Instruct + Thinking variants, Qwen3.5 hybrid backbone, 256K context, simplified vision merger.
  - Inference: [Single-image QA](v4.6/inference/single-image.html) · [Multi-image QA](v4.6/inference/multi-images.html) · [Video](v4.6/inference/video-understanding.html) · [OCR](v4.6/inference/ocr.html) · [PDF](v4.6/inference/pdf-parse.html) · [Grounding](v4.6/inference/grounding.html)
  - Deployment: [vLLM](v4.6/deployment/vllm.html) · [SGLang](v4.6/deployment/sglang.html) · [llama.cpp](v4.6/deployment/llamacpp.html) · [Ollama](v4.6/deployment/ollama.html)
  - Quantization: [GGUF](v4.6/quantization/gguf.html) · [BNB](v4.6/quantization/bnb.html) · [AWQ](v4.6/quantization/awq.html)

## Pick the right recipe

### Individuals

Effortless inference on your own machine — runs on **CPU + GPU**, **macOS / Linux / Windows**, even on phones.

- [Ollama](v4.6/deployment/ollama.html) — easiest setup
- [llama.cpp](v4.6/deployment/llamacpp.html) — fastest CPU inference
- [iOS demo](shared/demos/ios.html) — runs on iPhone / iPad

### Enterprises

High-throughput, scalable serving:

- [vLLM](v4.6/deployment/vllm.html) — production-grade GPU inference
- [SGLang](v4.6/deployment/sglang.html) — high-throughput GPU inference

### Researchers

Train / fine-tune / customize:

- [Transformers full / LoRA](shared/finetune/full.html)
- [LLaMA-Factory](shared/finetune/llamafactory.html)
- [SWIFT](shared/finetune/swift.html)
- [Align-anything](shared/finetune/align-anything.html)

## Versions

This cookbook tracks all currently supported MiniCPM releases:

### MiniCPM-V

| Version | Status | Modality | Size | Highlights | Context |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4.6** *(latest)* | Recommended | Image, Video | ~1.2B | Phone-ready edge MLLM, LLaVA-UHD v4 vision tower | 256K |
| 4.5 | Stable | Image, Video | 9B | Image + video understanding, optional thinking mode | 32K |

### MiniCPM-o

| Version | Status | Modality | Size | Highlights | Context |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4.5** *(latest)* | Recommended | Image, Video, Audio | 9B | End-to-end omnimodal (vision + speech + TTS), full-duplex streaming | 32K |

### MiniCPM (LLM)

| Version | Status | Modality | Size | Highlights | Context |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4.1** *(latest)* | Recommended | Text | 8B | Hybrid reasoning, EAGLE3, InfLLM-V2 | 128K |
| 4 | Stable | Text | 0.5B / 8B | InfLLM-V2, FRSpec speculative decoding | 128K |
| SALA | Research | Text | 8B | Sparse + linear hybrid attention | 1M+ |

Use the **version switcher** in the sidebar to jump between releases.

## Resources

- 🤗 [HuggingFace](https://huggingface.co/openbmb)
- 🤖 [ModelScope](https://modelscope.cn/organization/OpenBMB)
- 📖 [Technical Blog](https://huggingface.co/papers/2509.18154)
- 💬 [Discord](https://discord.gg/rftuRMbqzf)
- 🐛 [Open an Issue](https://github.com/OpenBMB/MiniCPM-o/issues)
