# AWQ

::::{Note}
**Support:** MiniCPM-V 4.7 (1B, 35B-A3B)
::::

> **1B only for a clean 4-bit checkpoint via AutoAWQ.** MiniCPM-V 4.7 **35B-A3B** keeps its MoE
> experts as fused parameters, which AutoAWQ cannot quantize (it only handles `nn.Linear`).
> For 35B-A3B use the **GGUF** variants or the published `-AWQ` / `-INT4` / `-FP8` checkpoints
> (see the [FP8/INT4 (llm-compressor)](../llmcompressor/minicpm-v4_7_llmcompressor_quantize.md) guide).

> The AutoAWQ workflow used by MiniCPM-V is hosted at [`tc-mb/AutoAWQ`](https://github.com/tc-mb/AutoAWQ)
> (the upstream `casper-hansen/AutoAWQ` is archived). The fork adds the
> `minicpmv4_7` architecture registration.

## Method 1 — Use the pre-quantized model with vLLM

### 1. Download the pre-quantized model

```bash
git clone https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-AWQ
```

### 2. Run with vLLM

```python
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_NAME = "openbmb/MiniCPM-V-4.7-1B-AWQ"
IMAGES = ["image.png"]

image = Image.open(IMAGES[0]).convert("RGB")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

llm = LLM(model=MODEL_NAME, max_model_len=8192, trust_remote_code=True)

messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "Please describe the content of this image"},
    ],
}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": image}},
    SamplingParams(temperature=0.0, max_tokens=1024),
)
print(outputs[0].outputs[0].text)
```

## Method 2 — Run the AWQ checkpoint with AutoAWQ directly

### 1. Build AutoAWQ from source

```bash
git clone https://github.com/tc-mb/AutoAWQ.git
cd AutoAWQ
pip install -e .
```

### 2. Inference script

```python
import torch
from PIL import Image
from transformers import AutoProcessor
from awq import AutoAWQForCausalLM

model_path = "openbmb/MiniCPM-V-4.7-1B-AWQ"
image_path = "./assets/airplane.jpeg"

model = AutoAWQForCausalLM.from_quantized(model_path, trust_remote_code=True).to("cuda")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

image = Image.open(image_path).convert("RGB")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "What is in this picture?"},
    ],
}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to("cuda")

with torch.inference_mode():
    out_ids = model.generate(**inputs, max_new_tokens=256)
print("Output:", processor.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Method 3 — Quantize the model yourself

### 1. Build AutoAWQ from source

```bash
git clone https://github.com/tc-mb/AutoAWQ.git
cd AutoAWQ
pip install -e .
```

### 2. Quantization script

```python
import os
import shutil
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

model_path = "openbmb/MiniCPM-V-4.7-1B"
quant_path = "./MiniCPM-V-4.7-1B-AWQ"

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
model.model.to("cuda:0")   # ensure the backbone is on GPU
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def copy_files_not_in_B(A_path, B_path):
    if not os.path.exists(B_path):
        os.makedirs(B_path)
    files_in_A = {f for f in os.listdir(A_path) if not (".bin" in f or "safetensors" in f)}
    for f in files_in_A - set(os.listdir(B_path)):
        src, dst = os.path.join(A_path, f), os.path.join(B_path, f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)


def load_alpaca():
    data = load_dataset("tatsu-lab/alpaca", split="train")

    def concatenate(x):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": x["instruction"]},
            {"role": "assistant", "content": x["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)}

    return [r["text"] for r in data.map(concatenate)][:1024]


calib_data = load_alpaca()
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
copy_files_not_in_B(model_path, quant_path)
print(f'Model is quantized and saved at "{quant_path}"')
```

A ready-to-run version with CLI arguments lives in
[`minicpm-v4_7_awq_quantize.py`](./minicpm-v4_7_awq_quantize.py):

```bash
python minicpm-v4_7_awq_quantize.py \
  --model-path openbmb/MiniCPM-V-4.7-1B \
  --save-path ./MiniCPM-V-4.7-1B-AWQ
```

## Result (1B)

| Item | Value |
|---|---|
| Quant method | `awq` (GEMM) |
| Bit width / group size | 4 / 128 |
| Zero point | yes |
| Size | ~2.3 GB (from 3.0 GB bf16) |
