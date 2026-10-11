# BNB

:::{Note}
**支持:** MiniCPM-V 4.7 (1B, 35B-A3B)
:::

> **只有 1B 能用 bitsandbytes 得到干净的 4-bit checkpoint。** MiniCPM-V 4.7 **35B-A3B** 的 MoE 专家以
> *融合参数* 存储（`mlp.experts.gate_up_proj`、`mlp.experts.down_proj`）。bitsandbytes 只量化
> `nn.Linear`，因此占权重约 92% 的专家张量仍保持 bf16，保存出来的 checkpoint 仍有 ~60 GB。
> 35B-A3B 要可用的 4-bit，请改用 **GGUF** 或已发布的 `-INT4` / `-FP8` 权重
> （见 [FP8/INT4（llm-compressor）](../llmcompressor/minicpm-v4_7_llmcompressor_quantize_zh.md)）。

## 1. 下载模型

从 [HuggingFace](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) 下载 MiniCPM-V 4.7
（MoE 变体为 `openbmb/MiniCPM-V-4.7-35B-A3B`），解压到本地目录。

## 2. 量化脚本

加载原始模型，用 bitsandbytes 量化为 NF4，跑一次推理自检，然后保存量化模型。

> 需要 `transformers>=5.2`（注册了 MiniCPM-V 4.7），并带 `trust_remote_code=True`。

```python
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

assert torch.cuda.is_available(), "需要 GPU。"

model_path = "openbmb/MiniCPM-V-4.7-1B"     # 原始模型
save_path = "./MiniCPM-V-4.7-1B-int4"       # 保存路径
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
        {"type": "text", "text": "这张图片里有什么？"},
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
print("量化后输出:", response)
print("量化后推理耗时:", time.time() - start)

os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path, safe_serialization=True)
processor.save_pretrained(save_path)
```

带命令行参数的即用版本见
[`minicpm-v4_7_bnb_quantize.py`](./minicpm-v4_7_bnb_quantize.py)：

```bash
python minicpm-v4_7_bnb_quantize.py \
  --model-path openbmb/MiniCPM-V-4.7-1B \
  --save-path ./MiniCPM-V-4.7-1B-int4
```

## 3. 预期输出

```text
[bnb] loaded in 1.5s
[bnb] sanity output (5.1s): 图中是一架在晴空中飞行的飞机……
[bnb] saving to ./MiniCPM-V-4.7-1B-int4
[bnb] done
```

量化模型保存在 `save_path`，可用
`AutoModelForImageTextToText.from_pretrained(save_path, trust_remote_code=True)` 重新加载。

## 4. 结果（1B）

| 项 | 值 |
|---|---|
| 位宽 | 4 (NF4, double quant) |
| `quant_method` | `bitsandbytes` |
| 体积 | ~1.5 GB（bf16 为 3.0 GB） |
