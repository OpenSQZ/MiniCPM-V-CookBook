# MiniCPM 4.1 - SGLang

> [!NOTE]
> SGLang upstream does not yet ship MiniCPM 4.1 support, so we maintain it on the [`tc-mb/sglang`](https://github.com/tc-mb/sglang) fork under the `minicpm` branch. The branch tracks SGLang `main` plus our MiniCPM 4 / 4.1 additions and is rebased regularly. We aim to upstream these changes once the API stabilises. (SALA lives on a separate [`minicpm_sala`](https://github.com/tc-mb/sglang/tree/minicpm_sala) branch.)

## 1. Environment

```bash
git clone https://github.com/tc-mb/sglang.git
cd sglang
git checkout minicpm

pip install --upgrade pip
pip install -e "python[all]"
```

> [!TIP]
> SGLang requires FlashInfer for the default attention backend. If installation fails, install it manually first:
>
> ```bash
> pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/
> ```

## 2. Launch the server

```bash
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM4.1-8B \
    --port 30000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --context-length 65536
```

The server exposes an OpenAI-compatible API on `http://localhost:30000/v1`.

## 3. Call the service

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")

resp = client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B",
    messages=[{"role": "user", "content": "Write a short article about edge AI."}],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### Hybrid reasoning over the API

```python
resp = client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B",
    messages=[{"role": "user", "content": "Solve: a train leaves at 9:00 traveling at 80 km/h to a city 200 km away..."}],
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
print(resp.choices[0].message.content)
```

## 4. Notes

- Branch in use: [`tc-mb/sglang @ minicpm`](https://github.com/tc-mb/sglang/tree/minicpm). Pinning a commit is recommended for reproducible deployments.
- For multi-GPU inference add `--tp 2` (tensor parallel size). MiniCPM 4.1 ships as a single 8B checkpoint so TP is usually unnecessary on modern GPUs.
- For audio / image inputs use MiniCPM-V or MiniCPM-o.
