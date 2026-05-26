# MiniCPM 4 — Overview

> The previous-generation MiniCPM LLM release. MiniCPM 4 introduced InfLLM-V2 sparse attention and remains supported alongside [MiniCPM 4.1](../minicpm4_1/overview.html).

## Checkpoints

| Variant | HuggingFace | ModelScope |
| :--- | :--- | :--- |
| Base / Instruct (8B) | [`openbmb/MiniCPM4-8B`](https://huggingface.co/openbmb/MiniCPM4-8B) | [`OpenBMB/MiniCPM4-8B`](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B) |
| Base / Instruct (0.5B) | [`openbmb/MiniCPM4-0.5B`](https://huggingface.co/openbmb/MiniCPM4-0.5B) | [`OpenBMB/MiniCPM4-0.5B`](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-0.5B) |
| Eagle (FRSpec, speculative decoding) | [`openbmb/MiniCPM4-8B-Eagle-FRSpec`](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-FRSpec) | [mirror](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-FRSpec) |
| Eagle FRSpec + QAT | [`openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT`](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT) | [mirror](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-FRSpec-QAT) |
| Eagle for vLLM | [`openbmb/MiniCPM4-8B-Eagle-vLLM`](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-vLLM) | [mirror](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-vLLM) |
| Marlin INT4 + Eagle for vLLM | [`openbmb/MiniCPM4-8B-marlin-Eagle-vLLM`](https://huggingface.co/openbmb/MiniCPM4-8B-marlin-Eagle-vLLM) | [mirror](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-marlin-Eagle-vLLM) |
| MiniCPM4-0.5B QAT (Int4) | [`openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format`](https://huggingface.co/openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format) | [mirror](https://modelscope.cn/models/OpenBMB/MiniCPM4-0.5B-QAT-Int4-GPTQ-format) |

## Highlights

- **InfLLM-V2 sparse attention** — long-context inference with under 5% of full-attention compute on a 128K window.
- **Eagle / FRSpec speculative decoding** — official draft checkpoints (including a QAT-friendly variant) for accelerated generation on the 8B model.
- **0.5B / 8B size points** — choose 0.5B for tightly constrained edge deployments, 8B for full reasoning capability.

## Quick start

### Inference (Hugging Face Transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM4-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
).eval().cuda()

messages = [{"role": "user", "content": "Write an article about Artificial Intelligence."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt",
).to(model.device)

out = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### Serve with vLLM

```bash
pip install -U vllm
vllm serve openbmb/MiniCPM4-8B --trust-remote-code --max-model-len 32768
```

## Where to next

- 🤗 [Model collection on HuggingFace](https://huggingface.co/collections/openbmb/minicpm-4-6841ab29d180257e940baa9b)
- 📖 [Technical report (arXiv 2506.07900)](https://arxiv.org/abs/2506.07900)
- 🛠️ [Main repository (OpenBMB/MiniCPM)](https://github.com/OpenBMB/MiniCPM)
- ➡️ For new projects we recommend [MiniCPM 4.1](../minicpm4_1/overview.html), which adds hybrid reasoning and EAGLE3 on top of this release.

> Per-feature guides (vLLM, SGLang, llama.cpp, Ollama, fine-tuning) are being added to this cookbook section. Use the version sidebar on the left to navigate as they land.
