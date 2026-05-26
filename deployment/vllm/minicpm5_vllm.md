# MiniCPM 5 - vLLM

> [!NOTE]
> MiniCPM 5 is supported on **upstream vLLM** with no fork. Because the model is a vanilla `LlamaForCausalLM`, vLLM `>= 0.21` loads the checkpoint directly — no custom kernels, no `trust_remote_code`. This is the recommended path for production-grade throughput and OpenAI-compatible chat completions.

## 1. Environment

```bash
pip install "vllm>=0.21"          # latest (CUDA 13.x driver hosts)
# pip install "vllm==0.10.1.1"    # fallback for CUDA 12.x driver hosts
```

> [!TIP]
> Use vLLM `>= 0.21` for first-class MiniCPM 5 support. Older versions predate the standard `LlamaForCausalLM` integration path.

## 2. API service

### Launch the server

```bash
vllm serve openbmb/MiniCPM5-1B \
    --served-model-name MiniCPM5-1B \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

The server exposes an OpenAI-compatible `/v1/chat/completions` endpoint on `http://localhost:8000` by default.

### Tuning knobs

| Flag | Default | When to change |
| :--- | :--- | :--- |
| `--max-model-len` | `131072` (native 128K) | drop to `8192` / `32768` to free KV-cache on small GPUs |
| `--gpu-memory-utilization` | `0.85` | drop on **shared** GPUs — vLLM hard-fails if `(free / total) < value` |
| `--dtype` | `bfloat16` | `float16` for older GPUs (newer NVIDIA prefers bf16) |
| `--enforce-eager` | unset | set if CUDA graphs OOM on tiny VRAM budgets |

### Call the service (OpenAI client)

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

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

Toggle `<think>` reasoning per-request via `extra_body.chat_template_kwargs.enable_thinking`:

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

### Quick sanity check (curl)

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "max_tokens": 64
    }'
```

## 3. Offline / batched inference

```python
from vllm import LLM, SamplingParams

llm = LLM(model="openbmb/MiniCPM5-1B", dtype="bfloat16", max_model_len=131072)
out = llm.chat(
    [[{"role": "user", "content": "用一句话解释 GQA。"}]],
    SamplingParams(temperature=0.9, top_p=0.95, max_tokens=512),
    chat_template_kwargs={"enable_thinking": True},
)
print(out[0].outputs[0].text)
```

## 4. Tool calling

For native `tool_calls` JSON output, prefer the [SGLang deployment guide](sglang.html) — SGLang ships a built-in `minicpm5` parser. With vLLM, MiniCPM 5's XML-style tool calls land inside the assistant's `content` string and need to be parsed client-side.

## 5. Notes

- MiniCPM 5 is text-only; for image / audio inputs use MiniCPM-V or MiniCPM-o.
- `--trust-remote-code` is **not required** — MiniCPM 5 ships as a standard `LlamaForCausalLM`.
- `--max-model-len` should not exceed the model's native 131,072 tokens. vLLM pre-allocates KV cache for the full context, so set it as low as your workload allows.
