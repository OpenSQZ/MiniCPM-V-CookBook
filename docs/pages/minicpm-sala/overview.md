# MiniCPM-SALA — Overview

> A research release exploring hybrid sparse-plus-linear attention for ultra-long-context inference. Distilled from MiniCPM 4 with structured decay and post-training adaptation.

## Checkpoints

| Variant | HuggingFace | ModelScope |
| :--- | :--- | :--- |
| MiniCPM-SALA | [`openbmb/MiniCPM-SALA`](https://huggingface.co/openbmb/MiniCPM-SALA) | [`OpenBMB/MiniCPM-SALA`](https://www.modelscope.cn/models/OpenBMB/MiniCPM-SALA) |

## What is SALA?

**SALA (Sparse Attention and Linear Attention)** is the first large-scale hybrid architecture that systematically combines:

- **25% sparse attention** ([InfLLM-V2](https://arxiv.org/abs/2509.24663)) — high-fidelity local modelling
- **75% linear attention** (Lightning Attention) — globally efficient recurrent computation

Further enhanced with [HyPE](https://arxiv.org/abs/2601.22156), SALA extends to million-token context windows while maintaining strong length generalization.

## Highlights

- **Up to 3.5× inference speedup** compared with Transformer baselines like Qwen3-8B in long-context settings.
- **Greatly reduced KV-cache footprint**, making million-token inputs practical on a single GPU.
- **Transformer → Hybrid distillation** — initialized from MiniCPM 4 with structured decay and post-training adaptation, preserving most of the dense model's capability.

## Quick start

### Inference (Hugging Face Transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM-SALA"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
).eval().cuda()

messages = [{"role": "user", "content": "Summarise this long document: ..."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt",
).to(model.device)

out = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### Serve with SGLang

```bash
# SALA needs the dedicated SGLang branch we maintain in tc-mb/sglang.
git clone https://github.com/tc-mb/sglang.git
cd sglang
git checkout minicpm_sala
pip install --upgrade pip
pip install -e "python[all]"

python -m sglang.launch_server \
    --model-path openbmb/MiniCPM-SALA \
    --port 30000 --trust-remote-code --dtype bfloat16
```

## Where to next

- 📖 [Technical report (in repo)](https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf)
- 🛠️ [Main repository (OpenBMB/MiniCPM)](https://github.com/OpenBMB/MiniCPM)
- 🔬 [InfLLM-V2 training kernels](https://github.com/OpenBMB/infllmv2_cuda_impl)

> Per-feature guides (Transformers chat, SGLang, LLaMA-Factory fine-tuning) are being added to this cookbook section. Use the version sidebar on the left to navigate as they land.
