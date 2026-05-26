# Dataset Preparation Guide

This tutorial will guide you on how to prepare datasets for supervised fine-tuning of MiniCPM-o. Please strictly follow the format and requirements below to organize your data.

## 1. Data Format Description

Each data sample should be a dictionary containing the image path and multi-turn conversation content. For example:

```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {"role": "user", "content": "Describe this image."},
    {"role": "assistant", "content": "This is a cat."}
  ]
}
```

- `image`: Image path, supports both single and multiple images (see below).
- `conversations`: Multi-turn conversation, must start with user, and role only supports "user" and "assistant".

### Multi-image Input Format

To support multiple images, the `image` field should be a dictionary, with keys like `<image_00>`, `<image_01>`, and values as image paths:

```json
{
  "image": {
    "<image_00>": "path/to/image1.jpg",
    "<image_01>": "path/to/image2.jpg"
  },
  "conversations": [
    {"role": "user", "content": "Compare <image_00> and <image_01>."},
    {"role": "assistant", "content": "The first image is a cat, the second is a dog."}
  ]
}
```

## 2. Dataset Loading and Preprocessing

Dataset loading and preprocessing mainly rely on the `SupervisedDataset` class. The core process is as follows:

- Read raw data (in json or list format).
- Load and transform images.
- Tokenize and encode conversation content to generate input_ids, labels, position_ids, etc.
- Support advanced image processing such as slicing and patching (slice_config).

### Main Parameter Description

- `raw_data`: List of raw data samples.
- `transform`: Image preprocessing method (e.g., normalization, resizing, etc.).
- `tokenizer`: Tokenizer, should match the model.
- `slice_config`: Image slicing configuration (optional).
- `llm_type`: Large model type (e.g., "minicpm", "llama3", "qwen").
- `patch_size`: Image patch size, default is 14.
- `query_nums`: Number of image tokens, default is 64.
- `batch_vision`: Whether to process images in batch, default is False.
- `max_length`: Maximum text length, default is 2048.

## 3. Dataset Examples

Single-image multi-turn conversation example:

```json
{
  "image": "images/cat.jpg",
  "conversations": [
    {"role": "user", "content": "<image> What animal is this?"},
    {"role": "assistant", "content": "This is a cat."},
    {"role": "user", "content": "What is it doing?"},
    {"role": "assistant", "content": "It is sleeping."}
  ]
}
```

Multi-image conversation example:

```json
{
  "image": {
    "<image_00>": "images/cat.jpg",
    "<image_01>": "images/dog.jpg"
  },
  "conversations": [
    {"role": "user", "content": "Compare <image_00> and <image_01>."},
    {"role": "assistant", "content": "The first image is a cat, the second is a dog."}
  ]
}
```

## 4. Common Issues and Notes

- The conversation must start with user, and role only supports "user" and "assistant".
- Image paths must be valid and support local paths.
- For multiple images, use `<image_xx>` placeholders in conversations to correspond to the image dictionary.
- For large images, it is recommended to configure `slice_config` for slicing.
- If data loading fails, the logger will automatically resample a data sample.

---
