# Official Fine-tuning Scripts

We provide official scripts for fine-tuning pretrained MiniCPM-V / MiniCPM-o models on downstream tasks, using **transformers Trainer** and **DeepSpeed**.

Supported models: **MiniCPM-V-4**, **MiniCPM-o-2_6**, **MiniCPM-V-2_6**, **MiniCPM-Llama3-V-2_5**, **MiniCPM-V-2**.

## 1. Data Preparation

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

### Dataset Loading and Preprocessing

Dataset loading and preprocessing mainly rely on the `SupervisedDataset` class. The core process is as follows:

- Read raw data (in json or list format).
- Load and transform images.
- Tokenize and encode conversation content to generate input_ids, labels, position_ids, etc.
- Support advanced image processing such as slicing and patching (slice_config).

#### Main Parameters

- `raw_data`: List of raw data samples.
- `transform`: Image preprocessing method (e.g., normalization, resizing, etc.).
- `tokenizer`: Tokenizer, should match the model.
- `slice_config`: Image slicing configuration (optional).
- `llm_type`: Large model type (e.g., "minicpm", "llama3", "qwen").
- `patch_size`: Image patch size, default is 14.
- `query_nums`: Number of image tokens, default is 64.
- `batch_vision`: Whether to process images in batch, default is False.
- `max_length`: Maximum text length, default is 2048.

### Common Issues and Notes

- The conversation must start with user, and role only supports "user" and "assistant".
- Image paths must be valid and support local paths.
- For multiple images, use `<image_xx>` placeholders in conversations to correspond to the image dictionary.
- For large images, it is recommended to configure `slice_config` for slicing.
- If data loading fails, the logger will automatically resample a data sample.

## 2. Full-parameter Fine-tuning

Full-parameter fine-tuning requires updating all parameters of LLM in the whole training process. Please specify the correct MODEL path, DATA path and LLM_TYPE in the shell scripts.

You can find and review the training script here: [finetune_ds.sh](./finetune_ds.sh)

```shell
MODEL="MiniCPM-o-2_6" # or "openbmb/MiniCPM-V-4", "openbmb/MiniCPM-V-2_6", "openbmb/MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-V-2"
DATA="path/to/trainging_data" # json file
EVAL_DATA="path/to/test_data" # json file
LLM_TYPE="qwen" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm, if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE="llama3",
# if use openbmb/MiniCPM-o-2_6 or openbmb/MiniCPM-V-2_6, please set LLM_TYPE=qwen
# if use openbmb/MiniCPM-V-4, please set LLM_TYPE=llama
```

To launch your training, run the following script:

```
sh finetune_ds.sh
```

## 3. LoRA Fine-tuning

LoRA allows light-weight model tuning with only a small subset of parameters updated. We provide the LoRA implementation based on `peft`. To launch your training, run the following script:

You can find and review the training script here: [finetune_lora.sh](./finetune_lora.sh)

```shell
MODEL="MiniCPM-o-2_6" # or "openbmb/MiniCPM-V-4", "openbmb/MiniCPM-V-2_6", "openbmb/MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-V-2"
DATA="path/to/trainging_data" # json file
EVAL_DATA="path/to/test_data" # json file
LLM_TYPE="qwen" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm, if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE="llama3",
# if use openbmb/MiniCPM-o-2_6 or openbmb/MiniCPM-V-2_6, please set LLM_TYPE=qwen
# if use openbmb/MiniCPM-V-4, please set LLM_TYPE=llama
```

To launch your training, run the following script:

```
sh finetune_lora.sh
```

## 4. Loading a Fine-tuned Model

After training (either full-parameter or LoRA), you can load the model with the path to the adapter. We advise you to use absolute path for your pretrained model, because LoRA only saves the adapter and the absolute path in the adapter configuration json file is used for finding out the pretrained model to load.

```python
from peft import PeftModel
from transformers import AutoModel

model_type = "openbmb/MiniCPM-o-2_6"  # or "openbmb/MiniCPM-V-4", "openbmb/MiniCPM-V-2_6", "openbmb/MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-V-2"
path_to_adapter = "path_to_your_fine_tuned_checkpoint"

model = AutoModel.from_pretrained(
    model_type,
    trust_remote_code=True,
)

lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True,
).eval().cuda()
```

## 5. Memory Usage Statistics

The following table presents the memory usage when fine-tuning using NVIDIA A100 (80 GiB) GPUs with DeepSpeed Zero-3, gradient checkpointing, and CPU offloading (max length 2048, batch size 1).

| Fine-tuning Method | GPUs: 2 | GPUs: 4 | GPUs: 8 |
|--------------------|---------|---------|---------|
| LoRA               | 14.4 GiB| 13.6 GiB| 13.1 GiB|
| Full-parameter     | 16.0 GiB| 15.8 GiB| 15.6 GiB|

Refer to [DeepSpeed Zero stages](https://huggingface.co/docs/transformers/deepspeed) to further reduce memory cost.
