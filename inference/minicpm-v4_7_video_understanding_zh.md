# 视频理解（MiniCPM-V 4.7）

> v4.7 的视频流程很简单：processor 逐帧处理，单视频的帧预算由 `max_slice_nums` 控制。
> 不需要特殊的视频后端。

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

## 视频 → 帧

```python
from decord import VideoReader, cpu

MAX_NUM_FRAMES = 32  # 小显存可调低

def encode_video(video_path: str):
    def uniform_sample(indices, n):
        gap = len(indices) / n
        return [indices[int(i * gap + gap / 2)] for i in range(n)]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # 约 1 fps
    frame_idx = list(range(0, len(vr), sample_fps))
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    return [Image.fromarray(f.astype("uint8")) for f in frames]
```

## 视频对话

```python
video_path = "assets/badminton.mp4"
frames = encode_video(video_path)
print("num frames:", len(frames))

messages = [{
    "role": "user",
    "content": [
        {"type": "video", "video": frames},
        {"type": "text",  "text":  "描述这段视频"},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    chat_template_kwargs={"enable_thinking": False},
    # 如需调整帧切片预算（越低越省显存）
    max_slice_nums=2,
).to(model.device)

out_ids = model.generate(**inputs, max_new_tokens=1024)
answer = processor.decode(
    out_ids[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print(answer)
```

> 若遇到显存不足：
> 1. 调低 `MAX_NUM_FRAMES`；
> 2. 调低 `max_slice_nums`（设为 1 则完全不做切片）；
> 3. 或使用 `bfloat16` + `attn_implementation="sdpa"`。
