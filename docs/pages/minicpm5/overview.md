# MiniCPM 5 — Overview

> The first release in the MiniCPM 5 series — a dense **1B** Transformer tuned for on-device, local deployment, and resource-constrained scenarios. Reaches 1B-class open-source SOTA on agentic tool use, code generation, and difficult reasoning, with one checkpoint serving both Think and No-Think modes.

## Checkpoints

| Variant | HuggingFace | ModelScope |
| :--- | :--- | :--- |
| **Final release (RL + OPD)** | [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) | [`OpenBMB/MiniCPM5-1B`](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B) |
| SFT-only checkpoint (before RL / OPD) | [`openbmb/MiniCPM5-1B-SFT`](https://huggingface.co/openbmb/MiniCPM5-1B-SFT) | [`OpenBMB/MiniCPM5-1B-SFT`](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-SFT) |
| Base checkpoint (pre-training only) | [`openbmb/MiniCPM5-1B-Base`](https://huggingface.co/openbmb/MiniCPM5-1B-Base) | [`OpenBMB/MiniCPM5-1B-Base`](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-Base) |
| GGUF *(llama.cpp / Ollama)* | [`openbmb/MiniCPM5-1B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF) | [`OpenBMB/MiniCPM5-1B-GGUF`](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-GGUF) |
| MLX *(Apple Silicon, 4-bit)* | [`openbmb/MiniCPM5-1B-MLX`](https://huggingface.co/openbmb/MiniCPM5-1B-MLX) | [`OpenBMB/MiniCPM5-1B-MLX`](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-1B-MLX) |

## What's new in MiniCPM 5

- **1B-class open-source SOTA.** Within a comparison set including **LFM2.5-1.2B-Thinking**, **Qwen3-0.6B**, and **Qwen3.5-0.8B**, MiniCPM5-1B reaches an average score of **42.57** across reasoning, knowledge, code, instruction-following, math, logic and agentic benchmarks — vs. **35.61** for the strongest baseline. Strengths are most visible on agentic tool use, code, and competition math.
- **Hybrid Reasoning, one checkpoint.** Built-in `<think>` chat template — switch between deliberate reasoning and fast assistant mode at request time via `enable_thinking=True/False`.
- **Standard `LlamaForCausalLM` architecture.** No custom kernels, no model-code fork, no `trust_remote_code`. Mainstream inference engines (Transformers / vLLM / SGLang / llama.cpp / Ollama / MLX) load the checkpoint directly.
- **Native long context.** 131,072-token context window, 16 / 2 GQA heads, 24 layers, 1.08B parameters (679M non-embedding).
- **Tool calling out of the box.** XML-style tool calls with a SGLang `minicpm5` parser that converts them to OpenAI-compatible `tool_calls` natively.
- **RL + OPD post-training.** ↑16 points average on math / code / instruction-following, ↓29 percentage points overlong-response rate vs. the SFT-only checkpoint.
- **Desktop Pet companion app.** [`OpenBMB/MiniCPM-Desk-Pet`](https://github.com/OpenBMB/MiniCPM-Desk-Pet) — local-LLM desktop pet driven by MiniCPM5-1B over a thin `llama.cpp` `llama-server` sidecar; Apple Silicon / NVIDIA / CPU paths, LoRA persona switching.

## Quick start

### Inference (Hugging Face Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openbmb/MiniCPM5-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [{"role": "user", "content": "Who are you? Please briefly introduce yourself."}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,    # set True for step-by-step reasoning
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

### Serve with vLLM

```bash
pip install "vllm>=0.21"
vllm serve openbmb/MiniCPM5-1B --port 8000
```

### Serve with SGLang (recommended for tool calling)

```bash
pip install "sglang[srt]>=0.5.12"
python -m sglang.launch_server --model-path openbmb/MiniCPM5-1B --port 30000 \
    --tool-call-parser minicpm5
```

### Run on edge devices (Ollama)

```bash
ollama run openbmb/minicpm5
```

## Recommended sampling

| Mode | `enable_thinking` | `temperature` | `top_p` | When to use |
| :--- | :---: | :---: | :---: | :--- |
| **Think** | `True` | 0.9 | 0.95 | reasoning, math, code, multi-step planning |
| **No-Think** | `False` | 0.7 | 0.95 | fast assistant, latency-bound chat |

`generation_config.json` is tuned for **Think** mode by default.

## Where to next

- 🤗 [Model collection on HuggingFace](https://huggingface.co/openbmb)
- 🛠️ [Main repository (OpenBMB/MiniCPM)](https://github.com/OpenBMB/MiniCPM)
- 📖 [MiniCPM 4 technical report (arXiv 2506.07900)](https://arxiv.org/abs/2506.07900) — RL + OPD methodology
- 🐱 [MiniCPM Desk Pet (companion app)](https://github.com/OpenBMB/MiniCPM-Desk-Pet)

> Per-feature guides (Chat, vLLM, SGLang, llama.cpp, Ollama, MLX, fine-tuning) are listed in the version sidebar on the left.
