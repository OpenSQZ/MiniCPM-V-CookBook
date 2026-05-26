# MiniCPM 4 - vLLM

> [!NOTE]
> MiniCPM 4 is supported on **upstream vLLM** — no fork required. The 8B / 0.5B base models, the [Eagle-vLLM](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-vLLM) speculative draft, and the [marlin-Eagle-vLLM](https://huggingface.co/openbmb/MiniCPM4-8B-marlin-Eagle-vLLM) INT4 + speculative bundle all run with the official `vllm` package.

## 1. Environment

```bash
pip install -U vllm
```

> [!TIP]
> Use vLLM `>= 0.9.0` for first-class MiniCPM 4 support.

## 2. Launch the server

```bash
vllm serve openbmb/MiniCPM4-8B \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9
```

For the ultra-light 0.5B variant:

```bash
vllm serve openbmb/MiniCPM4-0.5B \
    --trust-remote-code \
    --max-model-len 32768
```

OpenAI-compatible API is exposed at `http://localhost:8000/v1`.

## 3. Call the service

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

resp = client.chat.completions.create(
    model="openbmb/MiniCPM4-8B",
    messages=[{"role": "user", "content": "Write an article about Artificial Intelligence."}],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

## 4. Speculative decoding (Eagle-vLLM)

MiniCPM 4 ships an EAGLE-style draft head fine-tuned for vLLM at [`openbmb/MiniCPM4-8B-Eagle-vLLM`](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-vLLM):

```bash
vllm serve openbmb/MiniCPM4-8B \
    --trust-remote-code \
    --max-model-len 32768 \
    --speculative-config '{"method":"eagle","model":"openbmb/MiniCPM4-8B-Eagle-vLLM","num_speculative_tokens":5}'
```

## 5. Marlin INT4 + speculative

[`openbmb/MiniCPM4-8B-marlin-Eagle-vLLM`](https://huggingface.co/openbmb/MiniCPM4-8B-marlin-Eagle-vLLM) bundles Marlin INT4 weights with the matching Eagle draft for maximum throughput:

```bash
vllm serve openbmb/MiniCPM4-8B-marlin-Eagle-vLLM \
    --trust-remote-code \
    --max-model-len 32768 \
    --speculative-config '{"method":"eagle","model":"openbmb/MiniCPM4-8B-Eagle-vLLM","num_speculative_tokens":5}'
```

Marlin requires CUDA compute capability ≥ 8.0.

## 6. Offline inference

```python
from vllm import LLM, SamplingParams

llm = LLM(model="openbmb/MiniCPM4-8B", trust_remote_code=True, max_model_len=32768)
sampling = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=512)
outputs = llm.chat(
    [{"role": "user", "content": "Write an article about Artificial Intelligence."}],
    sampling_params=sampling,
)
print(outputs[0].outputs[0].text)
```

## 7. Notes

- MiniCPM 4 does **not** support the `enable_thinking` toggle; for hybrid reasoning use [MiniCPM 4.1](../minicpm4_1/deployment/vllm.html).
- `--trust-remote-code` is required.
- For vLLM speculative decoding pick the `MiniCPM4-8B-Eagle-vLLM` draft (the `*-Eagle-FRSpec*` variants are tuned for a different runtime and are **not** vLLM-compatible).
