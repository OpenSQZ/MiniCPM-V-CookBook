# 官方脚本微调

我们提供了基于 **transformers Trainer** 和 **DeepSpeed** 的官方微调脚本，方便在下游任务上微调预训练的 MiniCPM-V / MiniCPM-o 模型。

支持模型：**MiniCPM-V-4**、**MiniCPM-o-2_6**、**MiniCPM-V-2_6**、**MiniCPM-Llama3-V-2_5**、**MiniCPM-V-2**。

## 1. 数据集准备

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

### 数据集加载与预处理

数据加载和预处理主要依赖 `SupervisedDataset` 类。其核心流程如下：

- 读取原始数据（json 或 list 格式）。
- 对图片进行加载和 transform 处理。
- 对对话内容进行分词、编码，生成 input_ids、labels、position_ids 等。
- 支持图片切片、分块等高级处理（slice_config）。

#### 主要参数说明

- `raw_data`：原始数据列表。
- `transform`：图片预处理方法（如归一化、缩放等）。
- `tokenizer`：分词器，需与模型对应。
- `slice_config`：图片切片配置（可选）。
- `llm_type`：大模型类型（如 "minicpm", "llama3", "qwen"）。
- `patch_size`：图片分块大小，默认 14。
- `query_nums`：图片 token 数，默认 64。
- `batch_vision`：是否批量处理图片，默认 False。
- `max_length`：最大文本长度，默认 2048。

### 常见问题与注意事项

- 对话必须以 user 开头，且 role 仅支持 "user" 和 "assistant"。
- 图片路径需真实有效，支持本地路径。
- 多图时，conversations 中需用 `<image_xx>` 占位符与 image 字典对应。
- 若图片较大，建议配置 `slice_config` 进行切片处理。
- 若数据加载报错，日志会自动重采样一条数据。

## 2. 全参数微调

全参数微调会在整个训练过程中更新 LLM 的全部参数。请在 shell 脚本中正确设置 `MODEL`、`DATA` 与 `LLM_TYPE`。

训练脚本可在此查看：[finetune_ds.sh](./finetune_ds.sh)

```shell
MODEL="MiniCPM-o-2_6" # 或 "openbmb/MiniCPM-V-4"、"openbmb/MiniCPM-V-2_6"、"openbmb/MiniCPM-Llama3-V-2_5"、"openbmb/MiniCPM-V-2"
DATA="path/to/trainging_data" # json 文件
EVAL_DATA="path/to/test_data" # json 文件
LLM_TYPE="qwen" # 使用 openbmb/MiniCPM-V-2 时设为 minicpm；使用 openbmb/MiniCPM-Llama3-V-2_5 时设为 "llama3"；
# 使用 openbmb/MiniCPM-o-2_6 或 openbmb/MiniCPM-V-2_6 时设为 qwen；
# 使用 openbmb/MiniCPM-V-4 时设为 llama
```

启动训练：

```
sh finetune_ds.sh
```

## 3. LoRA 微调

LoRA 是轻量级的微调方法，只更新少量参数。我们基于 `peft` 提供了 LoRA 实现。

训练脚本可在此查看：[finetune_lora.sh](./finetune_lora.sh)

```shell
MODEL="MiniCPM-o-2_6" # 或 "openbmb/MiniCPM-V-4"、"openbmb/MiniCPM-V-2_6"、"openbmb/MiniCPM-Llama3-V-2_5"、"openbmb/MiniCPM-V-2"
DATA="path/to/trainging_data" # json 文件
EVAL_DATA="path/to/test_data" # json 文件
LLM_TYPE="qwen" # 使用 openbmb/MiniCPM-V-2 时设为 minicpm；使用 openbmb/MiniCPM-Llama3-V-2_5 时设为 "llama3"；
# 使用 openbmb/MiniCPM-o-2_6 或 openbmb/MiniCPM-V-2_6 时设为 qwen；
# 使用 openbmb/MiniCPM-V-4 时设为 llama
```

启动训练：

```
sh finetune_lora.sh
```

## 4. 加载微调模型

训练完成后（无论是全参数还是 LoRA），可通过 adapter 路径加载模型。建议为预训练模型使用绝对路径，因为 LoRA 仅保存 adapter，adapter config json 中的绝对路径用于定位预训练模型。

```python
from peft import PeftModel
from transformers import AutoModel

model_type = "openbmb/MiniCPM-o-2_6"  # 或 "openbmb/MiniCPM-V-4"、"openbmb/MiniCPM-V-2_6"、"openbmb/MiniCPM-Llama3-V-2_5"、"openbmb/MiniCPM-V-2"
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

## 5. 显存占用统计

下表为使用 NVIDIA A100 (80 GiB) GPU、DeepSpeed Zero-3、梯度检查点、CPU offload 条件下的显存占用（最大长度 2048，batch size 1）。

| 微调方式 | 2 卡 | 4 卡 | 8 卡 |
|---------|------|------|------|
| LoRA    | 14.4 GiB | 13.6 GiB | 13.1 GiB |
| 全参数   | 16.0 GiB | 15.8 GiB | 15.6 GiB |

参考 [DeepSpeed Zero stages](https://huggingface.co/docs/transformers/deepspeed) 进一步降低显存占用。
