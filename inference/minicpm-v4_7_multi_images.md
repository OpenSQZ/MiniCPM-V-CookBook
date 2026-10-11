# Multi Images (MiniCPM-V 4.7)

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

## Chat with multiple images

```python
image1 = Image.open("assets/multi1.png").convert("RGB")
image2 = Image.open("assets/multi2.png").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image1},
        {"type": "image", "image": image2},
        {"type": "text",  "text":  "Compare the two images, tell me about the differences between them."},
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

out_ids = model.generate(**inputs, max_new_tokens=512)
answer = processor.decode(
    out_ids[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print(answer)
```

## Sample images

![alt text](./assets/multi1.png)

![alt text](./assets/multi2.png)
