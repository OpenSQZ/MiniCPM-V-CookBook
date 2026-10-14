# PDF Parse (MiniCPM-V 4.7)

## Initialize model

```python
import torch
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

## PDF → images

```python
# requires poppler-utils:
#   sudo apt-get update && sudo apt-get install poppler-utils
from pdf2image import convert_from_path


def pdf_to_images(pdf_path: str, dpi: int = 200):
    images = convert_from_path(pdf_path, dpi=dpi)
    return [image.convert("RGB") for image in images]
```

## Usage example

```python
prompt = """
You are an OCR assistant. Your task is to identify and extract all visible text from the image provided. Preserve the original formatting as closely as possible, including:

- Line breaks and paragraphs
- Headings and subheadings
- Any tables, lists, bullet points, or numbered items
- Special characters, spacing, and alignment

Output strictly the extracted text in Markdown format, reflecting the layout and structure of the original image. Do not add commentary, interpretation, or summarization—only return the raw text content with its formatting.
"""

images = pdf_to_images("assets/parse.pdf")

content = [{"type": "image", "image": img} for img in images]
content.insert(0, {"type": "text", "text": prompt})

messages = [{"role": "user", "content": content}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    chat_template_kwargs={"enable_thinking": False},
).to(model.device)

out_ids = model.generate(**inputs, max_new_tokens=4096)
answer = processor.decode(
    out_ids[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print(answer)
```

> For long PDFs, process page-by-page (or a few pages at a time) to avoid OOM, and consider
> enabling thinking for layout-heavy pages.

## Sample PDF

![alt text](./assets/parse.png)

## Recommended sampling parameters

For instruct-mode generation we recommend: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.
