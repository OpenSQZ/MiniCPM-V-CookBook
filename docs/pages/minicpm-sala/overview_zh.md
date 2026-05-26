# MiniCPM-SALA — 概览

> 探索"稀疏 + 线性"混合注意力的研究版本，面向超长上下文推理。由 MiniCPM 4 通过结构化衰减与后训练适应蒸馏而来。

## 模型清单

| 变体 | HuggingFace | ModelScope |
| :--- | :--- | :--- |
| MiniCPM-SALA | [`openbmb/MiniCPM-SALA`](https://huggingface.co/openbmb/MiniCPM-SALA) | [`OpenBMB/MiniCPM-SALA`](https://www.modelscope.cn/models/OpenBMB/MiniCPM-SALA) |

## 什么是 SALA？

**SALA（Sparse Attention and Linear Attention）** 是首个大规模混合架构，系统地结合：

- **25% 稀疏注意力**（[InfLLM-V2](https://arxiv.org/abs/2509.24663)）— 高保真局部建模
- **75% 线性注意力**（Lightning Attention）— 全局高效循环计算

并通过 [HyPE](https://arxiv.org/abs/2601.22156) 进一步增强，可以扩展到百万令牌上下文，同时保持强大的长度泛化。

## 亮点

- **推理提速最高 3.5 倍**（相对 Qwen3-8B 等 Transformer 基线，长上下文场景）。
- **KV 缓存大幅压缩**，单卡即可处理百万令牌输入。
- **Transformer → 混合架构蒸馏** — 从 MiniCPM 4 初始化，经过结构化衰减与后训练适应，保留了密集模型的绝大部分能力。

## 快速开始

### 推理（Hugging Face Transformers）

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM-SALA"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
).eval().cuda()

messages = [{"role": "user", "content": "请总结这份长文档：..."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt",
).to(model.device)

out = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### 使用 SGLang 部署

```bash
# SALA 需要我们维护的 tc-mb/sglang 分支。
git clone https://github.com/tc-mb/sglang.git
cd sglang
git checkout minicpm_sala
pip install --upgrade pip
pip install -e "python[all]"

python -m sglang.launch_server \
    --model-path openbmb/MiniCPM-SALA \
    --port 30000 --trust-remote-code --dtype bfloat16
```

## 后续阅读

- 📖 [技术报告（仓库内 PDF）](https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf)
- 🛠️ [主仓库 OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM)
- 🔬 [InfLLM-V2 训练 kernel](https://github.com/OpenBMB/infllmv2_cuda_impl)

> 各功能的详细指南（Transformers chat、SGLang、LLaMA-Factory 微调）正在陆续加入本节。请通过左侧侧栏的版本切换查看。
