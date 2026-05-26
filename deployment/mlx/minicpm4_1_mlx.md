# MiniCPM 4.1 - MLX (Apple Silicon)

> [!NOTE]
> [`openbmb/MiniCPM4.1-8B-MLX`](https://huggingface.co/openbmb/MiniCPM4.1-8B-MLX) is the official Apple-Silicon-optimised release using [`mlx-lm`](https://github.com/ml-explore/mlx-lm). Designed for native Mac inference on M-series chips (M1 / M2 / M3 / M4).

## 1. Environment

Tested on macOS 14+ with Python 3.10+:

```bash
pip install -U mlx-lm
```

## 2. Generate from the CLI

```bash
mlx_lm.generate \
    --model openbmb/MiniCPM4.1-8B-MLX \
    --prompt "Write a short article about edge AI." \
    --max-tokens 512 \
    --temp 0.7 --top-p 0.8
```

## 3. Python API

```python
from mlx_lm import load, generate

model, tokenizer = load("openbmb/MiniCPM4.1-8B-MLX")

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a short article about edge AI."}],
    tokenize=False,
    add_generation_prompt=True,
)

response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=512,
    temp=0.7,
    top_p=0.8,
)
print(response)
```

### Hybrid reasoning

```python
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Solve: a train leaves at 9:00 traveling at 80 km/h..."}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=1024, temp=0.6, top_p=0.95))
```

## 4. Run as a server

`mlx-lm` ships an OpenAI-compatible server:

```bash
mlx_lm.server --model openbmb/MiniCPM4.1-8B-MLX --port 8000
```

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
print(client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B-MLX",
    messages=[{"role": "user", "content": "Hello."}],
).choices[0].message.content)
```

## 5. Notes

- MLX weights are quantised for Apple Silicon (4-bit by default). Memory footprint on M-series is around 5 GB.
- Performance scales with unified memory bandwidth — M3 Pro / M4 Pro and above give the best throughput.
- For non-Mac deployments use the standard vLLM / SGLang / llama.cpp guides instead.
