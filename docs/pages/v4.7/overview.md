# MiniCPM-V 4.7 — Overview

> The latest release in the MiniCPM-V series, in **two sizes**: an edge-friendly **1B** and a
> sparse Mixture-of-Experts **35B-A3B**.

## Models

| Checkpoint | Size | Language backbone | Context |
|---|---|---|---|
| [`openbmb/MiniCPM-V-4.7-1B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) | 1.55 B | `qwen3_5_text` (dense) | 256K |
| [`openbmb/MiniCPM-V-4.7-35B-A3B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) | 35.2 B total · ~3.5 B active | `qwen3_5_moe_text` (256 experts, 8/token) | 256K |

Both share the same SigLIP-based **canvas M-RoPE** vision tower (27 layers, patch 14,
mixed 4×/16× token compression) and support single-image, multi-image and video understanding.
The 1B model is the edge/on-device tier; the 35B-A3B model is the high-throughput serving tier.

## What's new in 4.7

- **Canvas M-RoPE.** Visual tokens are laid out on a 2-D canvas and each carries a
  `(temporal, height, width)` position, so slices of one image keep their spatial
  relationship and video frames keep their temporal order. This is the main delta over 4.6.
- **Two sizes, one architecture family.** A dense 1B and a 256-expert MoE 35B-A3B share the
  same vision tower and processor.
- **Standalone transformers architecture.** Registered as `MiniCPMV4_7ForConditionalGeneration`
  in **`transformers >= 5.18.0`**; the standard `AutoProcessor` + `AutoModelForImageTextToText`
  flow works **without** `trust_remote_code`.
- **Thinking is a request flag.** Unlike 4.6 (separate `-Thinking` checkpoints), 4.7 is a single
  checkpoint per size: enable the reasoning trace with `enable_thinking` in `chat_template_kwargs`.

## Quick start (HF Transformers)

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "openbmb/MiniCPM-V-4.7-1B"   # or "openbmb/MiniCPM-V-4.7-35B-A3B"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path, dtype="auto", device_map="auto",
).eval()

image = Image.open("example.jpg").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text",  "text":  "Describe the image."},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=256,
    # recommended instruct-mode sampling parameters
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=100,
    min_p=0.0,
    presence_penalty=0,
    repetition_penalty=1.03,
)
print(processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

### Thinking mode

The chat template enables a `<think>` block by default. To turn it off (recommended for
plain assistant replies), pass:

```python
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    chat_template_kwargs={"enable_thinking": False},
).to(model.device)
```

## Recommended sampling parameters

For instruct-mode generation we recommend: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.

## Deployment

| Framework | Status | Entry |
|---|---|---|
| [vLLM](deployment/vllm.html) | Supported upstream (PR [#58674](https://github.com/vllm-project/vllm/pull/58674)) | `vllm serve` |
| [SGLang](deployment/sglang.html) | Via fork [`tc-mb/sglang`](https://github.com/tc-mb/sglang) (branch `Support-MiniCPM-V4.7`); not upstream yet | `sglang.launch_server` |
| [llama.cpp](deployment/llamacpp.html) | Supported upstream (PR [#29416](https://github.com/ggml-org/llama.cpp/pull/29416)) | `llama-mtmd-cli` / `llama-server` |
| [Ollama](deployment/ollama.html) | Model code ready; runtime awaits the bundled llama.cpp catching up to #29416 | `ollama create -f` |

## Quantization

| Format | 1B | 35B-A3B | Repository suffix |
|---|:--:|:--:|---|
| GGUF | ✅ | ✅ | `-gguf` |
| BNB (int4) | ✅ | ✅ | `-int4` |
| AWQ | ✅ | ✅ | `-AWQ` |
| GPTQ | ✅ | ✅ | `-GPTQ` |
| FP8 | — | ✅ | `-FP8` (llm-compressor) |

- **GGUF** is the recommended portable / CPU-Metal format; see the [GGUF guide](quantization/gguf.html).
- The **35B-A3B** HF 4-bit checkpoints (`-AWQ` / `-GPTQ` / `-int4` / `-FP8`) are produced with
  fused-expert-aware tools (llm-compressor / GPTQModel); plain AutoAWQ/AutoGPTQ/bitsandbytes
  cannot quantize the fused MoE expert parameters.

## Model repositories

- Base: `MiniCPM-V-4.7-1B`, `MiniCPM-V-4.7-35B-A3B`
- Quantized: `-gguf`, `-int4`, `-AWQ`, `-GPTQ` (1B) · `-gguf`, `-FP8`, `-INT4`, `-AWQ`, `-GPTQ`, `-int4` (35B-A3B)
- Speculative decoding draft: `MiniCPM-V-4.7-1B-DSpark`, `MiniCPM-V-4.7-35B-A3B-DSpark`

## Where to next

- **Try it out:** [Single-image QA](inference/single-image.html), [Multi-image QA](inference/multi-images.html), [Video](inference/video-understanding.html)
- **Other visual tasks:** [OCR](inference/ocr.html), [PDF parsing](inference/pdf-parse.html), [Grounding](inference/grounding.html)
- **Deploy at scale:** [vLLM](deployment/vllm.html), [SGLang](deployment/sglang.html)
- **Run locally:** [llama.cpp](deployment/llamacpp.html), [Ollama](deployment/ollama.html)
- **Quantize:** [GGUF](quantization/gguf.html), [BNB](quantization/bnb.html), [AWQ](quantization/awq.html), [GPTQ](quantization/gptq.html)
