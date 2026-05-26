# MiniCPM 4 - SGLang

> [!NOTE]
> SGLang upstream does not yet ship MiniCPM 4 support, so we maintain it on the [`tc-mb/sglang`](https://github.com/tc-mb/sglang) fork under the `minicpm` branch. Same branch as MiniCPM 4.1 — pull once and you can swap between 4 and 4.1 freely. (SALA lives on a separate [`minicpm_sala`](https://github.com/tc-mb/sglang/tree/minicpm_sala) branch.)

## 1. Environment

```bash
git clone https://github.com/tc-mb/sglang.git
cd sglang
git checkout minicpm

pip install --upgrade pip
pip install -e "python[all]"
```

## 2. Launch the server

```bash
# 8B
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM4-8B \
    --port 30000 --trust-remote-code --dtype bfloat16 --context-length 32768

# 0.5B (edge-friendly)
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM4-0.5B \
    --port 30000 --trust-remote-code --dtype bfloat16 --context-length 32768
```

OpenAI-compatible API on `http://localhost:30000/v1`.

## 3. Call the service

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")

resp = client.chat.completions.create(
    model="openbmb/MiniCPM4-8B",
    messages=[{"role": "user", "content": "Write an article about Artificial Intelligence."}],
    temperature=0.7, top_p=0.8, max_tokens=512,
)
print(resp.choices[0].message.content)
```

## 4. Notes

- Branch in use: [`tc-mb/sglang @ minicpm`](https://github.com/tc-mb/sglang/tree/minicpm).
- MiniCPM 4 does not support the `enable_thinking` toggle. For hybrid reasoning use [MiniCPM 4.1](../minicpm4_1/deployment/sglang.html).
- For very long inputs see [MiniCPM-SALA](../minicpm-sala/deployment/sglang.html).
