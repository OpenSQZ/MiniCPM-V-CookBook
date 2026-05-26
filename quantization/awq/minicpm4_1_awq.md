# MiniCPM 4.1 - AWQ

> [!NOTE]
> Pre-quantized AWQ weights are published as [`openbmb/MiniCPM4.1-8B-AutoAWQ`](https://huggingface.co/openbmb/MiniCPM4.1-8B-AutoAWQ). For most users, downloading the pre-quantized model is enough — no calibration required.

## Method 1 — Use the pre-quantized model

### Download

```bash
git clone https://huggingface.co/openbmb/MiniCPM4.1-8B-AutoAWQ
```

Or on ModelScope: <https://www.modelscope.cn/models/OpenBMB/MiniCPM4.1-8B-AutoAWQ>

### Inference with vLLM

```bash
pip install -U vllm
vllm serve openbmb/MiniCPM4.1-8B-AutoAWQ --trust-remote-code --max-model-len 65536
```

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
print(client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B-AutoAWQ",
    messages=[{"role": "user", "content": "Write a short article about edge AI."}],
).choices[0].message.content)
```

AWQ in vLLM uses the [AWQ-Marlin](https://github.com/IST-DASLab/marlin) INT4 kernel by default on Ampere / Ada / Hopper, so throughput on the AWQ checkpoint matches or beats the pure-Marlin checkpoint on most workloads.

### Inference with Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM4.1-8B-AutoAWQ", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM4.1-8B-AutoAWQ",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()
```

> [!IMPORTANT]
> Transformers must have AWQ kernels installed: `pip install autoawq`. Transformers `>= 4.43` will auto-load the AWQ runtime.

## Method 2 — Quantize the model yourself

If you need a custom calibration set, you can run AWQ yourself.

### Install AutoAWQ

```bash
pip install autoawq
```

> Some MiniCPM variants need the [`tc-mb/AutoAWQ`](https://github.com/tc-mb/AutoAWQ) fork for kernel patches. For MiniCPM 4.1 the upstream `autoawq` is sufficient at the time of writing.

### Quantize

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path  = "openbmb/MiniCPM4.1-8B"
output_path = "./MiniCPM4.1-8B-AWQ"

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(output_path)
tokenizer.save_pretrained(output_path)
```

The output directory is a drop-in HF model: load it the same way as the official `openbmb/MiniCPM4.1-8B-AutoAWQ`.

## Notes

- AWQ Group Size `128`, W-bit `4`, ZP enabled — these are the settings that match the official release.
- For per-token-fast deployments on Ampere/Ada/Hopper, AWQ + Marlin kernel gives ~2× the throughput of FP16 at near-FP16 quality.
- The 0.5B model has no AWQ release; use a GGUF Q4 build via `llama.cpp` instead for tightly constrained edge deployments.
