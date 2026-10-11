# MiniCPM-V 4.7 — 概览

> MiniCPM-V 系列最新一代，提供**两个尺寸**：端侧友好的 **1B** 与稀疏 MoE 的 **35B-A3B**。

## 模型

| Checkpoint | 参数量 | 语言骨干 | 上下文 |
|---|---|---|---|
| [`openbmb/MiniCPM-V-4.7-1B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) | 1.55 B | `qwen3_5_text`（dense） | 256K |
| [`openbmb/MiniCPM-V-4.7-35B-A3B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) | 35.2 B 总 · ~3.5 B 激活 | `qwen3_5_moe_text`（256 专家，8/token） | 256K |

两者共用同一套 SigLIP **canvas M-RoPE** 视觉塔（27 层，patch 14，4×/16× 混合压缩），
均支持单图、多图与视频理解。1B 面向端侧/低资源，35B-A3B 面向高吞吐服务。

## 4.7 的新特性

- **Canvas M-RoPE。** 视觉 token 铺在二维 canvas 上，各带 `(时间, 高, 宽)` 位置，
  因此同一张图的切片保持空间关系、视频帧保持时间顺序。这是相对 4.6 的主要变化。
- **两尺寸、同架构族。** dense 1B 与 256-专家 MoE 的 35B-A3B 共用视觉塔与 processor。
- **原生 transformers 架构。** 注册为 `MiniCPMV4_7ForConditionalGeneration`，需
  **`transformers >= 5.18.0`**；标准 `AutoProcessor` + `AutoModelForImageTextToText`
  流程可用，**无需 `trust_remote_code`**。
- **Thinking 是请求开关。** 不同于 4.6 的独立 `-Thinking` 权重，4.7 每个尺寸只有一个 checkpoint，
  通过 `chat_template_kwargs` 里的 `enable_thinking` 开启思考。

## 快速开始（HF Transformers）

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "openbmb/MiniCPM-V-4.7-1B"   # 或 "openbmb/MiniCPM-V-4.7-35B-A3B"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path, dtype="auto", device_map="auto",
).eval()

image = Image.open("example.jpg").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text",  "text":  "描述这张图片。"},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

out = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

### 思考模式

对话模板默认开启 `<think>` 块。如需关闭（普通问答建议关闭）：

```python
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    chat_template_kwargs={"enable_thinking": False},
).to(model.device)
```

## 部署

| 框架 | 状态 | 入口 |
|---|---|---|
| [vLLM](deployment/vllm.html) | 上游已支持（PR [#58674](https://github.com/vllm-project/vllm/pull/58674)） | `vllm serve` |
| [SGLang](deployment/sglang.html) | 走 fork [`tc-mb/sglang`](https://github.com/tc-mb/sglang)（分支 `Support-MiniCPM-V4.7`）；上游暂未支持 | `sglang.launch_server` |
| [llama.cpp](deployment/llamacpp.html) | 上游已支持（PR [#29416](https://github.com/ggml-org/llama.cpp/pull/29416)） | `llama-mtmd-cli` / `llama-server` |
| [Ollama](deployment/ollama.html) | 模型代码已就绪；运行时待其内置 llama.cpp 同步到 #29416 | `ollama create -f` |

## 量化

| 格式 | 1B | 35B-A3B | 仓库后缀 |
|---|:--:|:--:|---|
| GGUF | ✅ | ✅ | `-gguf` |
| BNB (int4) | ✅ | ✅ | `-int4` |
| AWQ | ✅ | ✅ | `-AWQ` |
| GPTQ | ✅ | ✅ | `-GPTQ` |
| FP8 | — | ✅ | `-FP8`（llm-compressor） |

- **GGUF** 是推荐的通用 / CPU-Metal 格式，见 [GGUF 指南](quantization/gguf.html)。
- **35B-A3B** 的 HF 4-bit 权重（`-AWQ` / `-GPTQ` / `-int4` / `-FP8`）由支持融合专家的工具
  （llm-compressor / GPTQModel）产出；普通 AutoAWQ/AutoGPTQ/bitsandbytes 无法量化融合的 MoE 专家参数。

## 模型仓库

- 基座：`MiniCPM-V-4.7-1B`、`MiniCPM-V-4.7-35B-A3B`
- 量化：`-gguf`、`-int4`、`-AWQ`、`-GPTQ`（1B）· `-gguf`、`-FP8`、`-INT4`、`-AWQ`、`-GPTQ`、`-int4`（35B-A3B）
- 投机解码 draft：`MiniCPM-V-4.7-1B-DSpark`、`MiniCPM-V-4.7-35B-A3B-DSpark`

## 下一步

- **上手体验：** [单图问答](inference/single-image.html)、[多图问答](inference/multi-images.html)、[视频理解](inference/video-understanding.html)
- **其他视觉任务：** [OCR](inference/ocr.html)、[PDF 解析](inference/pdf-parse.html)、[视觉定位](inference/grounding.html)
- **规模化部署：** [vLLM](deployment/vllm.html)、[SGLang](deployment/sglang.html)
- **本地运行：** [llama.cpp](deployment/llamacpp.html)、[Ollama](deployment/ollama.html)
- **量化：** [GGUF](quantization/gguf.html)、[BNB](quantization/bnb.html)、[AWQ](quantization/awq.html)、[GPTQ](quantization/gptq.html)
