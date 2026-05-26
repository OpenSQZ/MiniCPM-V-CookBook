# MiniCPM 4.1 - vLLM

> [!NOTE]
> MiniCPM 4.1 is supported on **upstream vLLM** — no fork or branch switching required. The 8B base model, the [EAGLE3](https://huggingface.co/openbmb/MiniCPM4.1-8B-Eagle3) draft, and the [Marlin INT4](https://huggingface.co/openbmb/MiniCPM-4.1-8B-Marlin) variant all run with the official `vllm` package.

## 1. Environment

```bash
# Either install from PyPI (recommended)
pip install -U vllm

# Or build from source against the latest upstream
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install --editable . -v
```

> [!TIP]
> Use vLLM `>= 0.10.0`. Older versions predate MiniCPM 4.1 support.

## 2. API service

### Launch the server

```bash
vllm serve openbmb/MiniCPM4.1-8B \
    --trust-remote-code \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.9
```

The server exposes an OpenAI-compatible `/v1/chat/completions` endpoint on `http://localhost:8000` by default.

### Call the service (OpenAI client)

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

resp = client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B",
    messages=[
        {"role": "user", "content": "Write a short article about edge AI."}
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### Hybrid reasoning over the API

To enable step-by-step reasoning, pass `enable_thinking=True` via `extra_body.chat_template_kwargs`:

```python
resp = client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B",
    messages=[
        {"role": "user", "content": "If a train leaves at 9:00 traveling at 80 km/h to a city 200 km away, when does it arrive?"}
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
print(resp.choices[0].message.content)
```

## 3. Offline inference

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="openbmb/MiniCPM4.1-8B",
    trust_remote_code=True,
    max_model_len=65536,
    gpu_memory_utilization=0.9,
)

sampling = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=512)
outputs = llm.chat(
    [{"role": "user", "content": "Write a short article about edge AI."}],
    sampling_params=sampling,
)
print(outputs[0].outputs[0].text)
```

## 4. EAGLE3 speculative decoding

MiniCPM 4.1 ships an official EAGLE3 draft checkpoint, [`openbmb/MiniCPM4.1-8B-Eagle3`](https://huggingface.co/openbmb/MiniCPM4.1-8B-Eagle3), reported to deliver up to **3× faster generation** on reasoning workloads.

```bash
vllm serve openbmb/MiniCPM4.1-8B \
    --trust-remote-code \
    --max-model-len 65536 \
    --speculative-config '{"method":"eagle3","model":"openbmb/MiniCPM4.1-8B-Eagle3","num_speculative_tokens":5}'
```

The OpenAI-compatible API is unchanged — speculative decoding is transparent to clients.

## 5. Marlin INT4 acceleration

For Ampere / Ada / Hopper GPUs, the official [`openbmb/MiniCPM-4.1-8B-Marlin`](https://huggingface.co/openbmb/MiniCPM-4.1-8B-Marlin) INT4 weights cut latency further. Swap the model name on the server command — no other changes:

```bash
vllm serve openbmb/MiniCPM-4.1-8B-Marlin \
    --trust-remote-code \
    --max-model-len 65536
```

Marlin requires CUDA compute capability ≥ 8.0. On older GPUs use AWQ ([`openbmb/MiniCPM4.1-8B-AutoAWQ`](https://huggingface.co/openbmb/MiniCPM4.1-8B-AutoAWQ)) or GPTQ ([`openbmb/MiniCPM4.1-8B-GPTQ`](https://huggingface.co/openbmb/MiniCPM4.1-8B-GPTQ)) instead.

## 6. Notes

- `--trust-remote-code` is required because the modelling code is loaded from the Hub.
- Set `--max-model-len` only as high as you need — vLLM pre-allocates KV cache for the full context.
- For audio / image inputs use the multimodal MiniCPM-V or MiniCPM-o models; MiniCPM 4.1 is text-only.
