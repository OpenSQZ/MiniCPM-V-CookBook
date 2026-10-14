# 视频理解（MiniCPM-V 4.7）

> v4.7 的视频走标准 message 流程：在 message 的 content 里放一个 `{"type": "video", ...}`
> 条目，由 processor 自动解码 / 采样帧。帧预算与采样由 processor 参数控制 ——
> **无需手动解帧**，也不需要特殊的视频后端。

## 安装

```bash
pip install "transformers[torch]>=5.18.0" torchvision torchcodec
```

> **CUDA 提示：** `torchcodec`（视频解码）对 CUDA 版本较敏感。若遇到
> `RuntimeError: Could not load libtorchcodec`，可改用 **PyAV** ——
> `pip install "transformers[torch]>=5.18.0" torchvision av` ——
> 或在装 torch 时指定匹配的 CUDA（如 CUDA 12.8）：
> `pip install "transformers>=5.18.0" torchvision torchcodec --index-url https://download.pytorch.org/whl/cu128`

## 初始化模型

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "openbmb/MiniCPM-V-4.7-1B"  # 或 "openbmb/MiniCPM-V-4.7-35B-A3B"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)

# 视频场景建议用 Flash Attention 2，加速并省显存
# model = AutoModelForImageTextToText.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
```

## 视频对话

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/football.mp4"},
            {"type": "text", "text": "详细描述这段视频。按时间线展开，重点关注画面文字、界面变化、主要动作与场景切换。"},
        ],
    }
]

downsample_mode = "16x"  # 需要更精细的细节时用 "4x"

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

使用本地文件时直接传路径：

```python
{"type": "video", "url": "./assets/badminton.mp4"}
```

## 高级参数

传给 `apply_chat_template`（标注了 `generate` 的也需传给 `generate`）：

| 参数 | 默认 | 适用 | 说明 |
|------|------|------|------|
| `downsample_mode` | `"16x"` | 图像 & 视频 | 视觉 token 下采样。`"16x"` 合并 token 更高效；`"4x"` 保留 4 倍 token，细节更精细。**必须同时传给 `generate()`。** |
| `max_slice_nums` | `9` | 图像 & 视频 | 高分辨率图像切片的最大数量。越大保留细节越多。建议：图像 `36`，视频 `1`。 |
| `max_num_frames` | `128` | 仅视频 | 控制时间上下文长度、避免显存溢出。**短视频**（时长 ≤ `max_num_frames` 秒）：按 **1 FPS** 采样。**长视频**（> `max_num_frames` 秒）：在全时间线上均匀采样恰好 `max_num_frames` 帧。 |
| `stack_frames` | `1` | 仅视频 | 每秒采样点数量。`1` = 仅主帧。`N`（N>1）= 每秒 1 主帧 + N−1 子帧，子帧合成为一张网格图。短视频建议 `1`，长视频建议 `3`~`5`。 |
| `use_image_id` | `True` | 图像 & 视频 | 是否在每个图像/帧占位符前加 `<image_id>N</image_id>`。图像设 `True`，视频设 `False`。 |

> **注意：** `downsample_mode` 必须**同时**传给 `apply_chat_template`（决定占位符数量）和 `generate`（作用于视觉编码器）。其余参数只传给 `apply_chat_template` 即可。

## 推荐采样参数

在 instruct 模式下生成时，推荐使用：`temperature=0.7`、`top_p=0.95`、`top_k=100`、`min_p=0.0`、`presence_penalty=0`、`repetition_penalty=1.03`。
