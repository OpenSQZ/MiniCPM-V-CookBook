# 多图问答（MiniCPM-V 4.7）

## 初始化模型

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "openbmb/MiniCPM-V-4.7-1B"  # 或 "openbmb/MiniCPM-V-4.7-35B-A3B"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
).eval()
```

## 多图对话

```python
image1 = Image.open("assets/multi1.png").convert("RGB")
image2 = Image.open("assets/multi2.png").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image1},
        {"type": "image", "image": image2},
        {"type": "text",  "text":  "比较这两张图片，说明它们之间的差异。"},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    chat_template_kwargs={"enable_thinking": False},
).to(model.device)

out_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    # recommended instruct-mode sampling parameters
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=100,
    min_p=0.0,
    presence_penalty=0,
    repetition_penalty=1.03,
)
answer = processor.decode(
    out_ids[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print(answer)
```

## 示例图片

![alt text](./assets/multi1.png)

![alt text](./assets/multi2.png)

## 推荐采样参数

在 instruct 模式下生成时，推荐使用：`temperature=0.7`、`top_p=0.95`、`top_k=100`、`min_p=0.0`、`presence_penalty=0`、`repetition_penalty=1.03`。
