# 单图问答（MiniCPM-V 4.7）

> MiniCPM-V 4.7 已在 **`transformers >= 5.18.0`** 中注册为独立架构
> `MiniCPMV4_7ForConditionalGeneration`，标准 HuggingFace `Processor` + `model.generate`
> 流程即可使用，**无需 `trust_remote_code`**。

4.7 有两个尺寸，改 `model_path` 即可切换：

|`model_path`| 尺寸 |
|---|---|
| `openbmb/MiniCPM-V-4.7-1B` | dense 1B（端侧） |
| `openbmb/MiniCPM-V-4.7-35B-A3B` | MoE 35B-A3B（高吞吐） |

## 初始化模型

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "openbmb/MiniCPM-V-4.7-1B"   # 或 "openbmb/MiniCPM-V-4.7-35B-A3B"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",  # 或 "flash_attention_2"
).eval()
```

## 单图对话

```python
image = Image.open("./assets/single.png").convert("RGB")

# 第一轮
question = "图中是什么地貌？"
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

## 第二轮

```python
messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
messages.append({
    "role": "user",
    "content": [{"type": "text", "text": "去这里旅行需要注意什么？"}],
})

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

![alt text](./assets/single.png)

## 思考模式说明

4.7 每个尺寸只有一个 checkpoint：对话模板默认带 `<think>` 块，除非传入
`chat_template_kwargs={"enable_thinking": False}`。重 OCR / 推理类任务可保留思考；
普通问答建议关闭以获得更快的直答。

## 推荐采样参数

在 instruct 模式下生成时，推荐使用：`temperature=0.7`、`top_p=0.95`、`top_k=100`、`min_p=0.0`、`presence_penalty=0`、`repetition_penalty=1.03`。
