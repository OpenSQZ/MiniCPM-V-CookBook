# 使用 LLaMA-Factory 微调 MiniCPM 5

> [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 是社区使用最广的微调框架。**MiniCPM 5 本身就是标准 `LlamaForCausalLM`**，可以直接用 LLaMA-Factory 标准流程训练 —— 无需打补丁、无需修改 template registry。

## 安装

```bash
pip install "llamafactory==0.9.3"        # 或最新版 `pip install llamafactory`
```

## 1. 数据准备

LLaMA-Factory 通过 `dataset_info.json` 注册数据集。MiniCPM 5 的 chat template 适用 `formatting: sharegpt` + 标准 role tag：

```bash
mkdir -p ~/finetune_data
cd ~/finetune_data
```

`~/finetune_data/dataset_info.json`：

```json
{
  "my_chat_data": {
    "file_name": "my_chat_data.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

`my_chat_data.jsonl` 每行一条对话：

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## 2. LoRA SFT 配置

保存为 `lora_sft.yaml`：

```yaml
### model
model_name_or_path: openbmb/MiniCPM5-1B
trust_remote_code: false

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all          # 全部线性层；或 "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

### dataset
dataset: my_chat_data
dataset_dir: /path/to/finetune_data
template: empty           # MiniCPM 5 chat template 由 tokenizer_config.json 自动加载
cutoff_len: 4096
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: ./runs/minicpm5_lora
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
ddp_timeout: 180000000
```

> 💡 `template: empty` 让 LLaMA-Factory **直接使用 tokenizer 自带的 `chat_template.jinja`** —— 即 MiniCPM 5 的 ChatML 风格模板（带 Think / No-Think / tools）。**不要**用 `template: llama3` 或其他内置模板，会造成 token layout 错乱。

## 3. 训练

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train lora_sft.yaml
```

样例输出（200 条样本、1 epoch、单 GPU、bs=4、grad_acc=2、lr=2e-4）：

```text
{'loss': 4.1912, 'learning_rate': 0.000192, 'epoch': 0.2}
{'loss': 3.9248, 'learning_rate': 0.000150, 'epoch': 0.4}
{'loss': 3.8348, 'learning_rate': 0.000087, 'epoch': 0.6}
{'loss': 3.6936, 'learning_rate': 0.000029, 'epoch': 0.8}
{'loss': 3.6183, 'learning_rate': 0.000001, 'epoch': 1.0}
{'train_runtime': 11.45, 'train_samples_per_second': 17.5, 'train_loss': 3.85}
```

loss 单调下降 —— 框架 + chat template + tokenizer 全链路 OK。

## 4. 用 LoRA adapter 推理

adapter 会保存到 `output_dir/checkpoint-XXXX/` 与 `output_dir/`。合并部署：

```bash
llamafactory-cli export merge.yaml
```

`merge.yaml`：

```yaml
model_name_or_path: openbmb/MiniCPM5-1B
adapter_name_or_path: ./runs/minicpm5_lora
template: empty
finetuning_type: lora
export_dir: ./minicpm5-merged
export_size: 4
export_legacy_format: false
```

合并后是普通的 `LlamaForCausalLM`，可被任意部署后端（`vllm` / `sglang` / `transformers` / `llama.cpp` 的 GGUF）直接加载。

## 5. 全量 SFT（不用 LoRA）

显存够用（单卡 bf16 + AdamW 约 12 GB）时，去掉 `finetuning_type: lora` 与 LoRA 相关字段：

```yaml
finetuning_type: full
deepspeed: examples/deepspeed/ds_z2_config.json   # 可选，多 GPU 时启用
```

## 6. 其他框架

MiniCPM 5 上游仓库还提供其他框架的单页 cookbook，按需选用：

| 框架 | 说明 | 上游 cookbook |
| :--- | :--- | :--- |
| **TRL + PEFT** | 裸 Python，通过 `{% generation %}` 实现 assistant-only loss | [`trl.md`](https://github.com/OpenBMB/MiniCPM/blob/minicpm5/docs/finetune/trl.md) |
| **ms-swift** | ModelScope 工具链；**必须**传 `--model_type llama --template chatml` | [`ms_swift.md`](https://github.com/OpenBMB/MiniCPM/blob/minicpm5/docs/finetune/ms_swift.md) |
| **unsloth** | 单 GPU LoRA / QLoRA + 自定义 kernel | [`unsloth.md`](https://github.com/OpenBMB/MiniCPM/blob/minicpm5/docs/finetune/unsloth.md) |
| **xtuner** | InternLM 的 mmengine 流程；使用 `PROMPT_TEMPLATE.qwen_chat`（即 ChatML） | [`xtuner.md`](https://github.com/OpenBMB/MiniCPM/blob/minicpm5/docs/finetune/xtuner.md) |

## 常见问题

### `template not found` / 输入 token 错乱

很可能设了 `template: llama3` 之类。改用 `template: empty`，由模型自带的 `chat_template.jinja` 生效。

### `transformers >= 4.55 required`（同环境也装了 vLLM）

LLaMA-Factory 0.9.3 期望 `transformers==4.52`，vLLM 0.21 要求 `>=5.6`。建议拆两个虚拟环境：一个微调，一个服务。合并后的模型在两个环境间通用。

### 多 GPU

YAML 中加 `deepspeed: examples/deepspeed/ds_z2_config.json`，启动：

```bash
FORCE_TORCHRUN=1 llamafactory-cli train lora_sft.yaml
```

LLaMA-Factory 的 `examples/deepspeed/` 提供 ZeRO-2 / ZeRO-3 / ZeRO-3-offload 模板，均可直接用于 MiniCPM 5。
