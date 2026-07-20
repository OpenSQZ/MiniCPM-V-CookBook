# 临时 Omni 模型 vLLM 部署指南

> [!WARNING]
> 本文面向内部临时测试 checkpoint。模型名称、配置和使用方式后续可能变化，不应直接用于生产环境。

## 1. 当前支持范围

本适配基于 vLLM `main`，临时维护在：

- 仓库：<https://github.com/tc-mb/vllm>
- 分支：`tmp/omni`
- 已验证提交：`9d9efc137`

当前已验证：

- 文本生成
- 图片理解和 OCR
- 音频理解和多音频输入
- 4 卡 Tensor Parallel
- Qwen3.5 MoE MTP 投机解码
- FULL / PIECEWISE CUDA Graph

当前适配会加载 APM 和音频投影权重，用于 16 kHz 音频输入。TTS 权重仍不加载，本文不包含语音生成和实时双工能力。

## 2. 环境准备

建议使用独立的 Python 3.12 环境，避免其他包注册 vLLM 插件或覆盖 CUDA 依赖。

```bash
git clone --branch tmp/omni https://github.com/tc-mb/vllm.git
cd vllm

uv venv --python 3.12
source .venv/bin/activate

# 使用与当前源码提交匹配的预编译 CUDA 扩展，并安装音频依赖。
VLLM_USE_PRECOMPILED=1 uv pip install -e ".[audio]" --torch-backend=auto
```

确认版本和模型类可以导入：

```bash
python - <<'PY'
import torch
import transformers
import vllm
from vllm.model_executor.models.minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
)

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("model:", MiniCPMV4_6ForConditionalGeneration.__name__)
PY
```

设置临时 checkpoint 路径：

```bash
export MODEL_PATH=/path/to/omni-test-checkpoint
```

模型目录至少应包含 `config.json`、tokenizer、processor、chat template 和 safetensors 权重。配置应满足：

```text
architectures = ["MiniCPMV4_6ForConditionalGeneration"]
model_type = "minicpmv4_6"
text_config.model_type = "qwen3_5_moe_text"
mrope_mode = "canvas"
```

## 3. 启动 OpenAI 兼容服务

以下配置已在 4 张 RTX 4090 上验证。根据机器情况修改 `CUDA_VISIBLE_DEVICES`、`--tensor-parallel-size` 和显存比例。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve "$MODEL_PATH" \
  --served-model-name Omni-test \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --chat-template "$PWD/vllm/transformers_utils/chat_templates/template_minicpmv46_omni.jinja" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"audio":2,"image":2}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'
```

关键参数：

| 参数 | 说明 |
| :--- | :--- |
| `--trust-remote-code` | 加载临时 checkpoint 自带的 processor 和配置代码 |
| `--chat-template` | 使用支持 `input_audio` 的临时 Omni 模板，不修改 checkpoint |
| `--tensor-parallel-size 4` | 将 MoE 图文模型切分到 4 张 GPU |
| `--limit-mm-per-prompt` | 设置单请求允许的音频和图片数量 |
| `--max-model-len 8192` | 测试用上下文长度；增大时需要更多 KV cache |
| `--speculative-config` | 启用 checkpoint 内置的单层 MTP draft model |
| `--compilation-config` | 捕获常用 batch size 的 FULL / PIECEWISE CUDA Graph |

健康检查：

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

服务日志中应能看到：

```text
Canvas M-RoPE enabled
Profiling CUDA graph memory
Capturing CUDA graphs
Application startup complete
```

## 4. 文本请求

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Omni-test",
    "messages": [
      {"role": "user", "content": "请用一句话介绍你自己。"}
    ],
    "temperature": 0,
    "max_tokens": 128
  }'
```

## 5. 图片请求

OpenAI Python client 示例：

```python
import base64
import mimetypes

from openai import OpenAI


def image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="Omni-test",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url("test.jpg")},
                },
                {
                    "type": "text",
                    "text": "请描述图片中的物体、颜色和文字。",
                },
            ],
        }
    ],
    temperature=0,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

仅在可信环境中使用本地文件 URL。服务端必须显式放行目录：

```bash
vllm serve "$MODEL_PATH" \
  ... \
  --allowed-local-media-path /absolute/path/to/images
```

请求中的 URL 对应写成 `file:///absolute/path/to/images/test.jpg`。

## 6. 音频请求

音频输入采用 OpenAI `input_audio` 格式。服务端会将音频重采样到 16 kHz，并通过 Whisper-medium APM 编码后注入语言模型。

```python
import base64

from openai import OpenAI


def encode_audio(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="Omni-test",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encode_audio("speech.wav"),
                        "format": "wav",
                    },
                },
                {
                    "type": "text",
                    "text": "请转写这段音频，并概括主要内容。",
                },
            ],
        }
    ],
    temperature=0,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

多音频输入时，在 `content` 中连续添加多个 `input_audio` 项，并确保 `--limit-mm-per-prompt` 中的 `audio` 数量足够。

> [!NOTE]
> 音频输入与 TTS 输出是两条独立链路。当前支持理解 WAV、MP3 等输入音频，但响应仍是文本，不会返回生成语音。

## 7. 离线推理

```python
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL_PATH = "/path/to/omni-test-checkpoint"
IMAGE_PATH = Path("/absolute/path/to/test.jpg")

llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    tensor_parallel_size=4,
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
    compilation_config={
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    },
    allowed_local_media_path=str(IMAGE_PATH.parent),
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": IMAGE_PATH.as_uri()},
            },
            {"type": "text", "text": "请描述这张图片。"},
        ],
    }
]

outputs = llm.chat(
    messages,
    sampling_params=SamplingParams(temperature=0, max_tokens=256),
)
print(outputs[0].outputs[0].text)
```

## 8. 验证 MTP 是否生效

启动参数中包含 `--speculative-config` 后，服务会定期输出类似日志：

```text
SpecDecoding metrics:
Accepted: 50 tokens, Drafted: 76 tokens,
Avg Draft acceptance rate: 65.8%
```

只看到正常生成结果并不能证明 MTP 已启用，应同时检查该指标。

本次测试结果：

| 项目 | 结果 |
| :--- | :--- |
| 文本 API | HTTP 200，正常生成 |
| 图片 API | HTTP 200，正确识别形状、颜色和文字 |
| 单音频 API | HTTP 200，正确理解英文演讲内容 |
| 双音频 API | HTTP 200，正确判断两段音频内容相同 |
| MTP | 接受 50 / 76 draft tokens，接受率 65.8% |
| CUDA Graph | FULL / PIECEWISE 捕获成功 |

## 9. 常见问题

### `_moe_C::moe_sum()` 参数数量不匹配

典型错误：

```text
_moe_C::moe_sum() expected at most 2 argument(s) but received 4
```

Python 源码和 vLLM CUDA 扩展来自不同提交。回到 vLLM 仓库重新安装当前分支对应的预编译扩展：

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto --reinstall
```

### 未知权重 `apm` 或 `tts`

当前 checkout 不是 `tmp/omni` 最新适配，或者安装后 Python 仍从其他 vLLM 目录导入。检查：

```bash
python -c "import vllm; print(vllm.__file__, vllm.__version__)"
git rev-parse HEAD
```

### 本地图片被拒绝

错误包含：

```text
Cannot load local files without --allowed-local-media-path
```

启动服务时添加 `--allowed-local-media-path`，或改用 base64 data URL。

### 显存不足

依次尝试：

1. 降低 `--max-model-len`
2. 降低 `--gpu-memory-utilization`
3. 减少 `cudagraph_capture_sizes`
4. 临时添加 `--enforce-eager` 排查 CUDA Graph 显存问题

`--enforce-eager` 会关闭 CUDA Graph，因此只建议用于排查。
