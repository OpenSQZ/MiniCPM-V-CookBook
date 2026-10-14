# Single Image (MiniCPM-V 4.7)

> MiniCPM-V 4.7 is registered upstream in **`transformers >= 5.18.0`** as the standalone
> architecture `MiniCPMV4_7ForConditionalGeneration`, so the standard HuggingFace
> `Processor` + `model.generate` flow works out of the box — **no `trust_remote_code` needed**.

MiniCPM-V 4.7 comes in two sizes; swap `model_path` to pick one:

|`model_path`| Size |
|---|---|
| `openbmb/MiniCPM-V-4.7-1B` | dense 1B (edge) |
| `openbmb/MiniCPM-V-4.7-35B-A3B` | MoE 35B-A3B (high-throughput) |

## Initialize model

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "openbmb/MiniCPM-V-4.7-1B"   # or "openbmb/MiniCPM-V-4.7-35B-A3B"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",  # or "flash_attention_2"
).eval()
```

## Chat with a single image

```python
image = Image.open("./assets/single.png").convert("RGB")

# First round
question = "What is the landform in the picture?"
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
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

## Second round

```python
messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
messages.append({
    "role": "user",
    "content": [{"type": "text", "text": "What should I pay attention to when traveling here?"}],
})

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

## Sample image

![alt text](./assets/single.png)

## Notes on thinking

4.7 is a single checkpoint per size: the chat template emits a `<think>` block unless you pass
`chat_template_kwargs={"enable_thinking": False}`. Leave thinking on for hard OCR / reasoning
tasks; turn it off for snappy plain replies.

## Recommended sampling parameters

For instruct-mode generation we recommend: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.
