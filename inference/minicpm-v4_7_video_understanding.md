# Video Understanding (MiniCPM-V 4.7)

> v4.7 keeps the video pipeline simple: the processor is invoked frame-by-frame, and the
> per-video frame budget is controlled by `max_slice_nums`. No special video backend is required.

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

## Video → frames

```python
from decord import VideoReader, cpu

MAX_NUM_FRAMES = 32  # reduce if you OOM on small GPUs

def encode_video(video_path: str):
    def uniform_sample(indices, n):
        gap = len(indices) / n
        return [indices[int(i * gap + gap / 2)] for i in range(n)]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # roughly 1 fps
    frame_idx = list(range(0, len(vr), sample_fps))
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    return [Image.fromarray(f.astype("uint8")) for f in frames]
```

## Chat with video

```python
video_path = "assets/badminton.mp4"
frames = encode_video(video_path)
print("num frames:", len(frames))

messages = [{
    "role": "user",
    "content": [
        {"type": "video", "video": frames},
        {"type": "text",  "text":  "Describe the video"},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    chat_template_kwargs={"enable_thinking": False},
    # tune frame slice budget if needed (lower = less memory)
    max_slice_nums=2,
).to(model.device)

out_ids = model.generate(**inputs, max_new_tokens=1024)
answer = processor.decode(
    out_ids[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print(answer)
```

> If you hit GPU OOM:
> 1. lower `MAX_NUM_FRAMES`,
> 2. lower `max_slice_nums` (1 disables slicing entirely),
> 3. or run in `bfloat16` with `attn_implementation="sdpa"`.
