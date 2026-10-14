# OCR 文字识别（MiniCPM-V 4.7）

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

## 使用示例

```python
ocr_image = Image.open("./assets/ocr.png").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": ocr_image},
        {"type": "text",  "text":  "图片中的文字是什么？"},
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

out_ids = model.generate(**inputs, max_new_tokens=1024)
answer = processor.decode(
    out_ids[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print(answer)
```

## 示例图片

![alt text](./assets/ocr.png)

> 对中文密集或版面复杂的文本页，开启思考（`chat_template_kwargs={"enable_thinking": True}`）
> 往往能得到更忠实的转写，因为模型会先审视版面再作答；短文本/裁剪良好的图片可关闭思考以更快直答。

## 推荐采样参数

在 instruct 模式下生成时，推荐使用：`temperature=0.7`、`top_p=0.95`、`top_k=100`、`min_p=0.0`、`presence_penalty=0`、`repetition_penalty=1.03`。
