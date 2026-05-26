# MiniCPM-SALA - SGLang

> [!NOTE]
> SALA support lives on the [`tc-mb/sglang`](https://github.com/tc-mb/sglang) fork under the `minicpm_sala` branch (a dedicated branch separate from `minicpm` used by MiniCPM 4 / 4.1). It ships the hybrid sparse + linear attention kernels needed by SALA.

## 1. Environment

```bash
git clone https://github.com/tc-mb/sglang.git
cd sglang
git checkout minicpm_sala

pip install --upgrade pip
pip install -e "python[all]"
```

> [!IMPORTANT]
> SALA's sparse-attention layers depend on the [InfLLM-V2 CUDA kernels](https://github.com/OpenBMB/infllmv2_cuda_impl). Build & install them before launching the server:
>
> ```bash
> git clone https://github.com/OpenBMB/infllmv2_cuda_impl.git
> cd infllmv2_cuda_impl
> pip install -e .
> ```

## 2. Launch the server

```bash
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM-SALA \
    --port 30000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --context-length 131072
```

> [!TIP]
> SALA's main selling point is ultra-long context. Set `--context-length` to whatever you actually need — the linear-attention layers keep memory bounded, but KV cache for the sparse layers still scales with length.

## 3. Call the service

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")

with open("long_document.txt", "r", encoding="utf-8") as f:
    long_text = f.read()

resp = client.chat.completions.create(
    model="openbmb/MiniCPM-SALA",
    messages=[{
        "role": "user",
        "content": f"Below is a long document. Read it and answer the question.\n\n{long_text}\n\nQuestion: What is the main argument of section 3?"
    }],
    temperature=0.7, top_p=0.8, max_tokens=1024,
)
print(resp.choices[0].message.content)
```

## 4. Notes

- Branch in use: [`tc-mb/sglang @ minicpm_sala`](https://github.com/tc-mb/sglang/tree/minicpm_sala).
- Kernel dependency: [`OpenBMB/infllmv2_cuda_impl`](https://github.com/OpenBMB/infllmv2_cuda_impl). Required for the sparse-attention layers — without it the model will fall back to a slow reference path.
- SALA is a research checkpoint; expect APIs and behaviour to evolve.
