# MiniCPM 4.1 - MLX（Apple Silicon）

> [!NOTE]
> [`openbmb/MiniCPM4.1-8B-MLX`](https://huggingface.co/openbmb/MiniCPM4.1-8B-MLX) 是面向 Apple Silicon 优化的官方版本，使用 [`mlx-lm`](https://github.com/ml-explore/mlx-lm) 运行。专为 M 系列芯片（M1 / M2 / M3 / M4）上的原生 Mac 推理设计。

## 1. 环境准备

需要 macOS 14+，Python 3.10+：

```bash
pip install -U mlx-lm
```

## 2. 命令行生成

```bash
mlx_lm.generate \
    --model openbmb/MiniCPM4.1-8B-MLX \
    --prompt "写一篇关于端侧 AI 的短文。" \
    --max-tokens 512 \
    --temp 0.7 --top-p 0.8
```

## 3. Python API

```python
from mlx_lm import load, generate

model, tokenizer = load("openbmb/MiniCPM4.1-8B-MLX")

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "写一篇关于端侧 AI 的短文。"}],
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

### 混合思考

```python
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "求解：列车 9:00 出发，以 80 km/h 行驶到 200 km 外的城市..."}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=1024, temp=0.6, top_p=0.95))
```

## 4. 作为服务运行

`mlx-lm` 内置 OpenAI 兼容服务：

```bash
mlx_lm.server --model openbmb/MiniCPM4.1-8B-MLX --port 8000
```

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
print(client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B-MLX",
    messages=[{"role": "user", "content": "你好。"}],
).choices[0].message.content)
```

## 5. 注意事项

- MLX 权重已为 Apple Silicon 做了量化（默认 4-bit）。M 系列上的内存占用约 5 GB。
- 性能受统一内存带宽限制 —— M3 Pro / M4 Pro 及以上获得最佳吞吐。
- 非 Mac 部署请使用标准的 vLLM / SGLang / llama.cpp 指南。
