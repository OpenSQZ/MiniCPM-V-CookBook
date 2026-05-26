# MiniCPM 5 — 概览

> MiniCPM 5 系列的首个版本 —— 一款面向端侧、本地部署与资源受限场景调优的 **1B** 稠密 Transformer。在工具调用、代码生成与困难推理任务上达到 1B 级别开源 SOTA，同一权重可同时支持 Think / No-Think 两种行为。

## 模型清单

| 变体 | HuggingFace | ModelScope |
| :--- | :--- | :--- |
| **正式发布版（RL + OPD 后）** | [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) | [`OpenBMB/MiniCPM5-1B`](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B) |
| 仅 SFT（RL / OPD 前） | [`openbmb/MiniCPM5-1B-SFT`](https://huggingface.co/openbmb/MiniCPM5-1B-SFT) | [`OpenBMB/MiniCPM5-1B-SFT`](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-SFT) |
| Base（仅预训练） | [`openbmb/MiniCPM5-1B-Base`](https://huggingface.co/openbmb/MiniCPM5-1B-Base) | [`OpenBMB/MiniCPM5-1B-Base`](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-Base) |
| GGUF *（llama.cpp / Ollama）* | [`openbmb/MiniCPM5-1B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF) | [`OpenBMB/MiniCPM5-1B-GGUF`](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-GGUF) |
| MLX *（Apple Silicon, 4-bit）* | [`openbmb/MiniCPM5-1B-MLX`](https://huggingface.co/openbmb/MiniCPM5-1B-MLX) | [`OpenBMB/MiniCPM5-1B-MLX`](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-MLX) |

## MiniCPM 5 新特性

- **1B 级开源 SOTA。** 在 **LFM2.5-1.2B-Thinking**、**Qwen3-0.6B** 与 **Qwen3.5-0.8B** 等同级别强基线对比中，MiniCPM5-1B 在推理、知识、代码、指令遵循、数学、逻辑、Agent 等多类任务上的平均分达到 **42.57**，明显领先最强基线的 **35.61**。在工具调用、代码、竞赛数学上的优势最为显著。
- **混合思考，单一权重。** 通过 chat template 内置的 `<think>` 块与 `enable_thinking=True/False` 在请求级别切换深度推理 / 快速回答两种模式。
- **标准 `LlamaForCausalLM` 架构。** 无自定义 kernel、无 fork 模型代码，**无需 `trust_remote_code`**。Transformers / vLLM / SGLang / llama.cpp / Ollama / MLX 等主流引擎均可直接加载。
- **原生长上下文。** 131,072 token 上下文窗口；24 层、16 / 2 分组的 GQA、1.08B 参数（其中 679M 非嵌入参数）。
- **开箱即用的工具调用。** 模型输出 XML 风格的 tool call，SGLang 的 `minicpm5` parser 可原生转换为 OpenAI 兼容的 `tool_calls`。
- **RL + OPD 后训练。** 数学 / 代码 / 指令遵循平均分 ↑16 分，超长响应比例 ↓29 个百分点（相对 SFT-only 模型）。
- **桌面宠物配套应用。** [`OpenBMB/MiniCPM-Desk-Pet`](https://github.com/OpenBMB/MiniCPM-Desk-Pet) —— 由 MiniCPM5-1B 通过 `llama.cpp` `llama-server` sidecar 本地驱动的桌面宠物，支持 Apple Silicon / NVIDIA / CPU 路径与 LoRA 角色切换。

## 快速开始

### 推理（Hugging Face Transformers）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openbmb/MiniCPM5-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [{"role": "user", "content": "你是谁？请简要介绍一下你自己。"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,    # 设为 True 启用逐步推理
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

### 使用 vLLM 部署

```bash
pip install "vllm>=0.21"
vllm serve openbmb/MiniCPM5-1B --port 8000
```

### 使用 SGLang 部署（推荐用于工具调用）

```bash
pip install "sglang[srt]>=0.5.12"
python -m sglang.launch_server --model-path openbmb/MiniCPM5-1B --port 30000 \
    --tool-call-parser minicpm5
```

### 端侧运行（Ollama）

```bash
ollama run openbmb/minicpm5
```

## 推荐采样参数

| 模式 | `enable_thinking` | `temperature` | `top_p` | 适用场景 |
| :--- | :---: | :---: | :---: | :--- |
| **Think** | `True` | 0.9 | 0.95 | 推理、数学、代码、多步规划 |
| **No-Think** | `False` | 0.7 | 0.95 | 快速对话、延迟敏感场景 |

`generation_config.json` 默认按 **Think** 模式调优。

## 后续阅读

- 🤗 [HuggingFace 模型合集](https://huggingface.co/openbmb)
- 🛠️ [主仓库 OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM)
- 📖 [MiniCPM 4 技术报告（arXiv 2506.07900）](https://arxiv.org/abs/2506.07900) —— RL + OPD 方法论
- 🐱 [MiniCPM Desk Pet 桌面宠物](https://github.com/OpenBMB/MiniCPM-Desk-Pet)

> 各功能详细指南（Chat、vLLM、SGLang、llama.cpp、Ollama、MLX、微调）均已列入左侧版本侧栏。
