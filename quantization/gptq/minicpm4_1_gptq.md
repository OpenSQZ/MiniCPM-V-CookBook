# MiniCPM 4.1 - GPTQ (and QAT variant)

> [!NOTE]
> Pre-quantized GPTQ weights are available as [`openbmb/MiniCPM4.1-8B-GPTQ`](https://huggingface.co/openbmb/MiniCPM4.1-8B-GPTQ). The MiniCPM4-0.5B-QAT release is also distributed in **GPTQ-format storage**, so this guide covers both.

## Method 1 — Use the pre-quantized model

### Download

```bash
git clone https://huggingface.co/openbmb/MiniCPM4.1-8B-GPTQ
```

Or on ModelScope: <https://www.modelscope.cn/models/OpenBMB/MiniCPM4.1-8B-GPTQ>

### Inference with vLLM

```bash
pip install -U vllm
vllm serve openbmb/MiniCPM4.1-8B-GPTQ --trust-remote-code --max-model-len 65536
```

vLLM auto-selects the GPTQ-Marlin kernel on Ampere / Ada / Hopper for maximum throughput.

### Inference with Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM4.1-8B-GPTQ", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM4.1-8B-GPTQ",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()
```

> [!IMPORTANT]
> Transformers `>= 4.36` ships GPTQ runtime via `optimum`. Install with `pip install optimum auto-gptq`.

## Method 2 — Quantize the model yourself

```bash
git clone https://github.com/tc-mb/AutoGPTQ.git
cd AutoGPTQ
pip install -e .
```

> The `tc-mb/AutoGPTQ` fork contains the MiniCPM-specific config patches. We've upstreamed the model_type mapping to AutoGPTQ-NEXT — for new projects you can also try the official [`vllm-project/llm-compressor`](https://github.com/vllm-project/llm-compressor) which is now the recommended quantizer toolkit.

A complete quantization script is provided alongside this doc — see the [`tc-mb/AutoGPTQ` README](https://github.com/tc-mb/AutoGPTQ) for the recipe used for the official `MiniCPM4.1-8B-GPTQ` release.

## QAT variant

MiniCPM 4 ships an additional **quantization-aware training** variant: [`openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format`](https://huggingface.co/openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format).

- The QAT model is *trained* with quantization in the loop, so the INT4 weights recover more quality than post-training GPTQ.
- It is **physically stored in GPTQ format**, so the same vLLM / Transformers loading code works:

```bash
vllm serve openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format \
    --trust-remote-code --max-model-len 32768
```

```python
model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()
```

## Notes

- GPTQ in vLLM uses the GPTQ-Marlin kernel by default on capable GPUs; no extra flags needed.
- For MiniCPM 4 (non-4.1), the official GPTQ release is [`openbmb/MiniCPM4-8B-GPTQ`](https://huggingface.co/openbmb/MiniCPM4-8B-GPTQ) — same flow.
- For very memory-constrained deployments, prefer GGUF Q4 / Q3 via `llama.cpp` over GPTQ INT4.
