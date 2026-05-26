# MiniCPM 4.1 — 概览

> MiniCPM LLM 系列的最新版本，面向端侧推理优化，支持混合思考模式与 EAGLE3 投机解码。

## 模型清单

| 变体 | HuggingFace | ModelScope |
| :--- | :--- | :--- |
| Base / Instruct | [`openbmb/MiniCPM4.1-8B`](https://huggingface.co/openbmb/MiniCPM4.1-8B) | [`OpenBMB/MiniCPM4.1-8B`](https://www.modelscope.cn/models/OpenBMB/MiniCPM4.1-8B) |
| EAGLE3 draft（投机解码用） | [`openbmb/MiniCPM4.1-8B-Eagle3`](https://huggingface.co/openbmb/MiniCPM4.1-8B-Eagle3) | — |
| AutoAWQ INT4 | [`openbmb/MiniCPM4.1-8B-AutoAWQ`](https://huggingface.co/openbmb/MiniCPM4.1-8B-AutoAWQ) | [镜像](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-AutoAWQ) |
| GPTQ INT4 | [`openbmb/MiniCPM4.1-8B-GPTQ`](https://huggingface.co/openbmb/MiniCPM4.1-8B-GPTQ) | [镜像](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-GPTQ) |
| Marlin INT4 *（vLLM 优化）* | [`openbmb/MiniCPM-4.1-8B-Marlin`](https://huggingface.co/openbmb/MiniCPM-4.1-8B-Marlin) | [镜像](https://www.modelscope.cn/openbmb/MiniCPM-4.1-8B-Marlin) |
| GGUF *（llama.cpp / Ollama）* | [`openbmb/MiniCPM4.1-8B-GGUF`](https://huggingface.co/openbmb/MiniCPM4.1-8B-GGUF) | [镜像](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-GGUF) |
| MLX *（Apple Silicon）* | [`openbmb/MiniCPM4.1-8B-MLX`](https://huggingface.co/openbmb/MiniCPM4.1-8B-MLX) | [镜像](https://www.modelscope.cn/openbmb/MiniCPM4.1-8B-MLX) |

## 4.1 新特性

- **混合思考模式。** 通过 chat template 的 `enable_thinking=True/False` 在请求级别切换是否启用逐步推理 —— 单一权重，两种行为。
- **EAGLE3 投机解码。** 官方同步发布 draft 模型，在推理任务上提速 3 倍。参见 [MiniCPM4 技术报告](https://arxiv.org/abs/2506.07900)。
- **InfLLM-V2 稀疏注意力。** 128K 上下文场景下计算量不到全注意力的 5%。参见 [InfLLM-V2 论文](https://arxiv.org/abs/2509.24663)。
- **开箱即用的量化方案。** OpenBMB 直接提供 AWQ / GPTQ / Marlin / GGUF / MLX 官方权重，常见部署场景无需自行量化。

## 快速开始

### 推理（Hugging Face Transformers）

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM4.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
).eval().cuda()

messages = [{"role": "user", "content": "写一篇关于端侧 AI 的短文。"}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False,   # 设为 True 启用逐步推理
).to(model.device)

out = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### 使用 vLLM 部署

```bash
# MiniCPM4.1 已合入 vLLM 上游 main 分支，无需 fork。
pip install -U vllm

vllm serve openbmb/MiniCPM4.1-8B --trust-remote-code --max-model-len 65536
```

### 端侧运行（Ollama）

```bash
ollama run openbmb/minicpm4.1
```

## 后续阅读

- 🤗 [HuggingFace 模型合集](https://huggingface.co/collections/openbmb/minicpm-4-6841ab29d180257e940baa9b)
- 📖 [技术报告（arXiv 2506.07900）](https://arxiv.org/abs/2506.07900)
- 🛠️ [主仓库 OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM)

> 各功能的详细指南（vLLM、SGLang、llama.cpp、Ollama、MLX、量化、微调）正在陆续加入本节。请通过左侧侧栏的版本切换查看。
