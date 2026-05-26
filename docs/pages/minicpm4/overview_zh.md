# MiniCPM 4 — 概览

> MiniCPM 上一代 LLM 系列模型。MiniCPM 4 引入了 InfLLM-V2 稀疏注意力，与 [MiniCPM 4.1](../minicpm4_1/overview.html) 同时维护。

## 模型清单

| 变体 | HuggingFace | ModelScope |
| :--- | :--- | :--- |
| Base / Instruct（8B） | [`openbmb/MiniCPM4-8B`](https://huggingface.co/openbmb/MiniCPM4-8B) | [`OpenBMB/MiniCPM4-8B`](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B) |
| Base / Instruct（0.5B） | [`openbmb/MiniCPM4-0.5B`](https://huggingface.co/openbmb/MiniCPM4-0.5B) | [`OpenBMB/MiniCPM4-0.5B`](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-0.5B) |
| Eagle（FRSpec，投机解码） | [`openbmb/MiniCPM4-8B-Eagle-FRSpec`](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-FRSpec) | [镜像](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-FRSpec) |
| Eagle FRSpec + QAT | [`openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT`](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT) | [镜像](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-FRSpec-QAT) |
| Eagle for vLLM | [`openbmb/MiniCPM4-8B-Eagle-vLLM`](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-vLLM) | [镜像](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-Eagle-vLLM) |
| Marlin INT4 + Eagle for vLLM | [`openbmb/MiniCPM4-8B-marlin-Eagle-vLLM`](https://huggingface.co/openbmb/MiniCPM4-8B-marlin-Eagle-vLLM) | [镜像](https://www.modelscope.cn/models/OpenBMB/MiniCPM4-8B-marlin-Eagle-vLLM) |
| MiniCPM4-0.5B QAT（Int4） | [`openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format`](https://huggingface.co/openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format) | [镜像](https://modelscope.cn/models/OpenBMB/MiniCPM4-0.5B-QAT-Int4-GPTQ-format) |

## 亮点

- **InfLLM-V2 稀疏注意力** — 128K 上下文场景下，计算量不到全注意力的 5%。
- **Eagle / FRSpec 投机解码** — 8B 模型配套的 draft 模型（含 QAT 友好变体），用于加速生成。
- **0.5B / 8B 双尺寸** — 0.5B 用于资源受限的端侧场景，8B 用于完整的推理能力。

## 快速开始

### 推理（Hugging Face Transformers）

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM4-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
).eval().cuda()

messages = [{"role": "user", "content": "写一篇关于人工智能的文章。"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt",
).to(model.device)

out = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### 使用 vLLM 部署

```bash
pip install -U vllm
vllm serve openbmb/MiniCPM4-8B --trust-remote-code --max-model-len 32768
```

## 后续阅读

- 🤗 [HuggingFace 模型合集](https://huggingface.co/collections/openbmb/minicpm-4-6841ab29d180257e940baa9b)
- 📖 [技术报告（arXiv 2506.07900）](https://arxiv.org/abs/2506.07900)
- 🛠️ [主仓库 OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM)
- ➡️ 新项目推荐使用 [MiniCPM 4.1](../minicpm4_1/overview.html)，在本版本基础上加入了混合思考与 EAGLE3。

> 各功能的详细指南（vLLM、SGLang、llama.cpp、Ollama、微调）正在陆续加入本节。请通过左侧侧栏的版本切换查看。
