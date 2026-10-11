# 视觉定位（MiniCPM-V 4.7）

## 示例代码

```python
import re
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText


def setup_model_and_processor(model_path: str):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    return model, processor


def extract_bbox_from_response(response: str):
    match = re.search(r"<box>([\d\s]+)</box>", response)
    if not match:
        raise ValueError("Can't find bbox in response")
    return list(map(int, match.group(1).strip().split()))


def draw_bbox_on_image(image: Image.Image, bbox):
    w, h = image.size
    x1 = int(bbox[0] / 1000 * w)
    y1 = int(bbox[1] / 1000 * h)
    x2 = int(bbox[2] / 1000 * w)
    y2 = int(bbox[3] / 1000 * h)
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    return image


def model_infer_and_draw(img_path, question, model, processor):
    image = Image.open(img_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  question},
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

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=1024)

    response = processor.decode(
        out_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    print("Model output:", response)

    bbox = extract_bbox_from_response(response)
    return response, bbox, draw_bbox_on_image(image, bbox)


model_path = "openbmb/MiniCPM-V-4.7-1B"  # 或 "openbmb/MiniCPM-V-4.7-35B-A3B"
img_path = "./assets/airplane.jpeg"
question = "Please provide the bounding box coordinate of the region this sentence describes: <ref>airplane</ref>"

model, processor = setup_model_and_processor(model_path)
response, bbox, image_with_box = model_infer_and_draw(img_path, question, model, processor)
image_with_box.save("./assets/airplane_grounding.jpeg")
```

## 效果展示

### 飞机

原图：

![alt text](./assets/airplane.jpeg)

定位后：

![alt text](./assets/airplane-grounding.jpeg)

### 多目标

多目标定位时，让模型返回 JSON（而非 `<box>`）再解析：

```python
import json
import re
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText


def extract_bboxes_from_json_response(response: str):
    response_json = json.loads(response)
    bboxes, names = [], []
    for item in response_json:
        bbox_str = re.sub(r"</?box>", "", item.get("box", "")).strip()
        bboxes.append(list(map(int, bbox_str.split())))
        names.append((item.get("name_en", ""), item.get("name_zh", "")))
    return bboxes, names


def draw_multiple_bboxes(image: Image.Image, bboxes, names):
    w, h = image.size
    draw = ImageDraw.Draw(image)
    for (x1, y1, x2, y2), (en, zh) in zip(bboxes, names):
        x1 = int(x1 / 1000 * w); y1 = int(y1 / 1000 * h)
        x2 = int(x2 / 1000 * w); y2 = int(y2 / 1000 * h)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, y1 - 10), f"{en} / {zh}", fill="red")
    return image


model_path = "openbmb/MiniCPM-V-4.7-1B"
img_path = "./assets/doraemon.jpg"
question = (
    "Identify all characters in the image and return their bounding boxes "
    "and English name and Chinese name in JSON format, "
    'a list of {"name_en": "Doraemon", "name_zh": "哆啦A梦", "box": "x1 y1 x2 y2"}'
)

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path, dtype="auto", device_map="auto", attn_implementation="sdpa",
).eval()

image = Image.open(img_path).convert("RGB")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  question},
    ],
}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
    chat_template_kwargs={"enable_thinking": False},
).to(model.device)

with torch.inference_mode():
    out_ids = model.generate(**inputs, max_new_tokens=1024)

response = processor.decode(
    out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
)
bboxes, names = extract_bboxes_from_json_response(response)
draw_multiple_bboxes(image, bboxes, names).save("./assets/doraemon_multi_grounding.jpg")
```

原图：

![alt text](./assets/doraemon.jpg)

定位后：

![alt text](./assets/doraemon_multi_grounding.jpg)
