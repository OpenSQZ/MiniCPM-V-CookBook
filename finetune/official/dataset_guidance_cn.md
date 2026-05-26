# 数据集准备指南

本教程将指导你如何为 MiniCPM-o 进行有监督微调准备数据集。请严格按照以下格式和要求组织你的数据。

## 1. 数据格式说明

每条数据应为一个字典，包含图片路径和多轮对话内容。例如：

```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {"role": "user", "content": "请描述这张图片"},
    {"role": "assistant", "content": "这是一只猫。"}
  ]
}
```

- `image`：图片路径，支持单图和多图（见下文）。
- `conversations`：多轮对话，必须以 user 开头，role 仅支持 "user" 和 "assistant"。

### 多图输入格式

如需支持多图输入，`image` 字段应为字典，键为 `<image_00>`, `<image_01>` 等，值为图片路径：

```json
{
  "image": {
    "<image_00>": "path/to/image1.jpg",
    "<image_01>": "path/to/image2.jpg"
  },
  "conversations": [
    {"role": "user", "content": "请比较 <image_00> 和 <image_01> 的不同"},
    {"role": "assistant", "content": "图片一是猫，图片二是狗。"}
  ]
}
```

## 2. 数据集加载与预处理

数据加载和预处理主要依赖 `SupervisedDataset` 类。其核心流程如下：

- 读取原始数据（json 或 list 格式）。
- 对图片进行加载和 transform 处理。
- 对对话内容进行分词、编码，生成 input_ids、labels、position_ids 等。
- 支持图片切片、分块等高级处理（slice_config）。

### 主要参数说明

- `raw_data`：原始数据列表。
- `transform`：图片预处理方法（如归一化、缩放等）。
- `tokenizer`：分词器，需与模型对应。
- `slice_config`：图片切片配置（可选）。
- `llm_type`：大模型类型（如 "minicpm", "llama3", "qwen"）。
- `patch_size`：图片分块大小，默认 14。
- `query_nums`：图片 token 数，默认 64。
- `batch_vision`：是否批量处理图片，默认 False。
- `max_length`：最大文本长度，默认 2048。

## 3. 数据集示例

单图多轮对话示例：

```json
{
  "image": "images/cat.jpg",
  "conversations": [
    {"role": "user", "content": "<image> 这是什么动物？"},
    {"role": "assistant", "content": "这是一只猫。"},
    {"role": "user", "content": "它在做什么？"},
    {"role": "assistant", "content": "它正在睡觉。"}
  ]
}
```

多图对话示例：

```json
{
  "image": {
    "<image_00>": "images/cat.jpg",
    "<image_01>": "images/dog.jpg"
  },
  "conversations": [
    {"role": "user", "content": "请比较 <image_00> 和 <image_01> 的不同"},
    {"role": "assistant", "content": "图片一是猫，图片二是狗。"}
  ]
}
```

## 4. 常见问题与注意事项

- 对话必须以 user 开头，且 role 仅支持 "user" 和 "assistant"。
- 图片路径需真实有效，支持本地路径。
- 多图时，conversations 中需用 `<image_xx>` 占位符与 image 字典对应。
- 若图片较大，建议配置 `slice_config` 进行切片处理。
- 若数据加载报错，日志会自动重采样一条数据。


---
