# OCR (MiniCPM-V 4.7)

## Initialize model

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "openbmb/MiniCPM-V-4.7-1B"  # or "openbmb/MiniCPM-V-4.7-35B-A3B"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
).eval()
```

## Usage example

```python
ocr_image = Image.open("./assets/ocr.png").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": ocr_image},
        {"type": "text",  "text":  "What is the text in the picture?"},
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
    max_new_tokens=1024,
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

## Sample image

![alt text](./assets/ocr.png)

> For Chinese-heavy or dense text pages, enabling thinking (`chat_template_kwargs={"enable_thinking": True}`)
> often produces more faithful transcriptions because the model inspects the layout before answering.
> Turn it off for short / well-cropped text to get faster answers.

## Recommended sampling parameters

For instruct-mode generation we recommend: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.
