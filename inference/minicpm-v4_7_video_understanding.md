# Video Understanding (MiniCPM-V 4.7)

> v4.7 handles video through the normal message flow: put a `{"type": "video", ...}`
> entry in the message content and let the processor decode / sample frames. Frame
> budget and sampling are controlled by processor kwargs — no manual frame decoding
> and no special video backend required.

## Install

```bash
pip install "transformers[torch]>=5.18.0" torchvision torchcodec
```

> **CUDA note:** `torchcodec` (video decoding) can be picky about CUDA versions.
> If you hit `RuntimeError: Could not load libtorchcodec`, either
> **use `PyAV` instead** — `pip install "transformers[torch]>=5.18.0" torchvision av` —
> or pin the CUDA build when installing torch (e.g. CUDA 12.8):
> `pip install "transformers>=5.18.0" torchvision torchcodec --index-url https://download.pytorch.org/whl/cu128`

## Initialize model

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "openbmb/MiniCPM-V-4.7-1B"  # or "openbmb/MiniCPM-V-4.7-35B-A3B"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)

# Flash Attention 2 is recommended for better acceleration and memory saving,
# especially in video scenarios.
# model = AutoModelForImageTextToText.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
```

## Chat with video

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/football.mp4"},
            {"type": "text", "text": "Describe this video in detail. Follow the timeline and focus on on-screen text, interface changes, main actions, and scene changes."},
        ],
    }
]

downsample_mode = "16x"  # use "4x" for finer detail

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
    downsample_mode=downsample_mode,
    max_num_frames=128,
    stack_frames=1,
    max_slice_nums=1,
    use_image_id=False,
).to(model.device)

generated_ids = model.generate(**inputs, downsample_mode=downsample_mode, max_new_tokens=2048)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
```

To use a local file instead of a URL, pass the path directly:

```python
{"type": "video", "url": "./assets/badminton.mp4"}
```

## Advanced parameters

Pass these to `apply_chat_template` (and, where noted, to `generate`):

| Parameter | Default | Applies to | Description |
|-----------|---------|------------|-------------|
| `downsample_mode` | `"16x"` | Image & Video | Visual token downsampling. `"16x"` merges tokens for efficiency; `"4x"` keeps 4× more tokens for finer detail. **Must also be passed to `generate()`.** |
| `max_slice_nums` | `9` | Image & Video | Maximum number of slices when splitting a high-resolution image. Higher values preserve more detail. Recommended: `36` for image, `1` for video. |
| `max_num_frames` | `128` | Video only | Controls the temporal context length and prevents VRAM overflow. **Short videos** (≤ `max_num_frames` sec): sampled at **1 FPS**. **Long videos** (> `max_num_frames` sec): uniformly sampled to exactly `max_num_frames` frames. |
| `stack_frames` | `1` | Video only | Sample points per second. `1` = main frame only. `N` (N>1) = 1 main frame + N−1 sub-frames per second, composited into a grid image. Recommended `1` for short videos, `3`–`5` for long videos. |
| `use_image_id` | `True` | Image & Video | Whether to prepend `<image_id>N</image_id>` before each image/frame placeholder. Set `True` for image, `False` for video. |

> **Note:** `downsample_mode` must be passed to **both** `apply_chat_template` (for the correct placeholder count) and `generate` (for the vision encoder). All other parameters only need to go to `apply_chat_template`.

## Recommended sampling parameters

For instruct-mode generation we recommend: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.
