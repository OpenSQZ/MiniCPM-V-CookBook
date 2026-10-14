# BNB

:::{Note}
**Support:** MiniCPM-V 4.7 (1B, 35B-A3B)
:::

> **1B only for a clean 4-bit checkpoint via bitsandbytes.** MiniCPM-V 4.7 **35B-A3B** stores
> its MoE experts as *fused* parameters (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`).
> bitsandbytes only quantizes `nn.Linear` modules, so the expert tensors (~92% of the weights)
> stay in bf16 and the saved checkpoint is still ~60 GB. For a usable 4-bit 35B-A3B use the
> **GGUF** variants or the published `-INT4` / `-FP8` checkpoints (see the
> [FP8/INT4 (llm-compressor)](../llmcompressor/minicpm-v4_7_llmcompressor_quantize.md) guide).

## 1. Download the model

Download MiniCPM-V 4.7 from [HuggingFace](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)
(or `openbmb/MiniCPM-V-4.7-35B-A3B` for the MoE variant) and extract it to a local directory.

## 2. Quantization script

Loads the original model, quantizes it to NF4 with bitsandbytes, runs a sanity-check
inference, and saves the quantized model.

> Requires `transformers>=5.2` (where MiniCPM-V 4.7 is registered) with
> `trust_remote_code=True`.

```python
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

assert torch.cuda.is_available(), "CUDA is not available, but this code requires a GPU."

model_path = "openbmb/MiniCPM-V-4.7-1B"     # original model
save_path = "./MiniCPM-V-4.7-1B-int4"       # where to save
image_path = "./assets/airplane.jpeg"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.uint8,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["lm_head"],
    llm_int8_threshold=6.0,
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True,
)

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
).to(model.device)

start = time.time()
with torch.inference_mode():
    out_ids = model.generate(**inputs, max_new_tokens=256)
response = processor.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("Output after quantization:", response)
print("Inference time after quantization:", time.time() - start)

os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path, safe_serialization=True)
processor.save_pretrained(save_path)
```

A ready-to-run version with CLI arguments lives in
[`minicpm-v4_7_bnb_quantize.py`](./minicpm-v4_7_bnb_quantize.py):

```bash
python minicpm-v4_7_bnb_quantize.py \
  --model-path openbmb/MiniCPM-V-4.7-1B \
  --save-path ./MiniCPM-V-4.7-1B-int4
```

## 3. Expected output

```text
[bnb] loaded in 1.5s
[bnb] sanity output (5.1s): The image depicts an aircraft in mid-flight against a clear blue sky. ...
[bnb] saving to ./MiniCPM-V-4.7-1B-int4
[bnb] done
```

The quantized model is saved at `save_path` and reloads with
`AutoModelForImageTextToText.from_pretrained(save_path, trust_remote_code=True)`.

## 4. Result (1B)

| Item | Value |
|---|---|
| Bit width | 4 (NF4, double quant) |
| `quant_method` | `bitsandbytes` |
| Size | ~1.5 GB (from 3.0 GB bf16) |

## Recommended sampling parameters

For instruct-mode generation we recommend: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.
