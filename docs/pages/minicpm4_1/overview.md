# MiniCPM 4.1 — Overview

> The latest release of the MiniCPM LLM series, tuned for edge-device inference with hybrid reasoning and EAGLE3 speculative decoding.

## Checkpoints

| Variant | HuggingFace | ModelScope |
| :--- | :--- | :--- |
| Base / Instruct | [`openbmb/MiniCPM4.1-8B`](https://huggingface.co/openbmb/MiniCPM4.1-8B) | [`OpenBMB/MiniCPM4.1-8B`](https://www.modelscope.cn/models/OpenBMB/MiniCPM4.1-8B) |
| EAGLE3 draft (speculative decoding) | [`openbmb/MiniCPM4.1-8B-Eagle3`](https://huggingface.co/openbmb/MiniCPM4.1-8B-Eagle3) | — |
| AutoAWQ INT4 | [`openbmb/MiniCPM4.1-8B-AutoAWQ`](https://huggingface.co/openbmb/MiniCPM4.1-8B-AutoAWQ) | [mirror](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-AutoAWQ) |
| GPTQ INT4 | [`openbmb/MiniCPM4.1-8B-GPTQ`](https://huggingface.co/openbmb/MiniCPM4.1-8B-GPTQ) | [mirror](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-GPTQ) |
| Marlin INT4 *(vLLM-optimised)* | [`openbmb/MiniCPM-4.1-8B-Marlin`](https://huggingface.co/openbmb/MiniCPM-4.1-8B-Marlin) | [mirror](https://www.modelscope.cn/openbmb/MiniCPM-4.1-8B-Marlin) |
| GGUF *(llama.cpp / Ollama)* | [`openbmb/MiniCPM4.1-8B-GGUF`](https://huggingface.co/openbmb/MiniCPM4.1-8B-GGUF) | [mirror](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-GGUF) |
| MLX *(Apple Silicon)* | [`openbmb/MiniCPM4.1-8B-MLX`](https://huggingface.co/openbmb/MiniCPM4.1-8B-MLX) | [mirror](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-MLX) |

## What's new in 4.1

- **Hybrid reasoning mode.** Toggle step-by-step reasoning at request time via `enable_thinking=True/False` on the chat template — one checkpoint, two behaviours.
- **EAGLE3 speculative decoding.** Official draft checkpoint shipped alongside the base model, delivering 3× faster generation on reasoning tasks. See the [MiniCPM4 paper](https://arxiv.org/abs/2506.07900).
- **InfLLM-V2 sparse attention.** 128K-token context with under 5% of full-attention compute. See the [InfLLM-V2 paper](https://arxiv.org/abs/2509.24663).
- **Full quantization stack out of the box.** Official AWQ / GPTQ / Marlin / GGUF / MLX weights from OpenBMB — no need to quantize yourself for common deployment targets.

## Quick start

### Inference (Hugging Face Transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM4.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
).eval().cuda()

messages = [{"role": "user", "content": "Write a short article about edge AI."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False,   # set True to enable step-by-step reasoning
).to(model.device)

out = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### Serve with vLLM

```bash
# MiniCPM4.1 is supported on upstream vLLM main; no fork required.
pip install -U vllm

vllm serve openbmb/MiniCPM4.1-8B --trust-remote-code --max-model-len 65536
```

### Run on edge devices with Ollama

```bash
ollama run openbmb/minicpm4.1
```

## Where to next

- 🤗 [Model collection on HuggingFace](https://huggingface.co/collections/openbmb/minicpm-4-6841ab29d180257e940baa9b)
- 📖 [Technical report (arXiv 2506.07900)](https://arxiv.org/abs/2506.07900)
- 🛠️ [Main repository (OpenBMB/MiniCPM)](https://github.com/OpenBMB/MiniCPM)

> Per-feature guides (vLLM, SGLang, llama.cpp, Ollama, MLX, quantization, fine-tuning) are being added to this cookbook section. Use the version sidebar on the left to navigate as they land.
