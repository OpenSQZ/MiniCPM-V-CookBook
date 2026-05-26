# MiniCPM 5 - SGLang

> [!NOTE]
> [SGLang](https://github.com/sgl-project/sglang) `>= 0.5.12` ships first-class MiniCPM 5 support, including a built-in **`minicpm5` tool-call parser** that converts the model's XML-style tool calls to OpenAI-compatible `tool_calls` natively. SGLang serves MiniCPM 5 as a standard `LlamaForCausalLM` with **RadixAttention prefix cache**, very high concurrency, and an OpenAI-compatible API. **This is the recommended backend for tool / function calling.**

## 1. Environment

```bash
pip install "sglang[srt]>=0.5.12"          # latest, requires CUDA 13.x driver
# pip install "sglang==0.5.6.post3"        # fallback for CUDA 12.x drivers
```

> [!TIP]
> SGLang requires FlashInfer for the default attention backend. If installation fails, install it manually first:
>
> ```bash
> pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/
> ```

Recommended runtime env vars:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1
```

## 2. Launch the server

```bash
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM5-1B \
    --served-model-name MiniCPM5-1B \
    --dtype bfloat16 \
    --context-length 131072 \
    --mem-fraction-static 0.85 \
    --tool-call-parser minicpm5 \
    --host 0.0.0.0 \
    --port 30000
```

The server exposes an OpenAI-compatible API on `http://localhost:30000/v1`.

### Tuning knobs

| Flag | Default | When to change |
| :--- | :--- | :--- |
| `--context-length` | `131072` (native 128K) | drop on small / shared GPUs |
| `--mem-fraction-static` | `0.85` | drop on shared GPUs |
| `--dtype` | `bfloat16` | use `float16` on Ampere or older |
| `--tool-call-parser` | unset | set to `minicpm5` (or `auto`) for native tool calls |

## 3. Call the service

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")

resp = client.chat.completions.create(
    model="MiniCPM5-1B",
    messages=[{"role": "user", "content": "Write a short article about edge AI."}],
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### Hybrid reasoning over the API

```python
resp = client.chat.completions.create(
    model="MiniCPM5-1B",
    messages=[{"role": "user", "content": "鸡兔同笼，头共 10 个，脚共 28 只，问鸡和兔各几只？"}],
    temperature=0.9,
    top_p=0.95,
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
print(resp.choices[0].message.content)
```

| Mode | `enable_thinking` | `temperature` | `top_p` |
| :--- | :---: | :---: | :---: |
| Think | `true` | 0.9 | 0.95 |
| No-Think | `false` | 0.7 | 0.95 |

## 4. Tool calling (recommended)

MiniCPM 5 emits XML-style tool calls. With `--tool-call-parser minicpm5`, SGLang converts them to OpenAI-compatible `tool_calls` JSON automatically — drop-in for any OpenAI tool-using client.

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B",
        "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
        "tool_choice": "auto",
        "temperature": 0.7,
        "max_tokens": 256
    }'
```

A typical response includes a structured `tool_calls` field:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Beijing\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

> [!TIP]
> `--tool-call-parser auto` will also auto-detect MiniCPM 5 by chat template and pick the `minicpm5` parser. Pin `--tool-call-parser minicpm5` explicitly for reproducible deployments.

## 5. Offline / batched inference (Engine API)

```python
import sglang as sgl

llm = sgl.Engine(
    model_path="openbmb/MiniCPM5-1B",
    tp_size=1,
    mem_fraction_static=0.8,
    context_length=131072,
)

outputs = llm.generate(
    ["用一句话解释什么是 GQA。"],
    sampling_params={
        "temperature": 0.9, "top_p": 0.95, "max_new_tokens": 1024,
        "skip_special_tokens": False,
    },
)
print(outputs)
```

## 6. Notes

- MiniCPM 5 is supported on **upstream SGLang `>= 0.5.12`** — no fork required.
- The `minicpm5` tool-call parser is a SGLang built-in; no extra packages.
- For multi-GPU, add `--tp 2` (tensor parallel size). On modern GPUs a single 1B-class checkpoint rarely needs TP > 1.
