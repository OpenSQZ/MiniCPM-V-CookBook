# LLaMA-Factory (MiniCPM 4 / 4.1)

> [!NOTE]
> [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) is a one-stop framework for SFT / DPO / RLHF fine-tuning. MiniCPM 4 / 4.1 ride the upstream LLaMA-Factory mainline — no fork required.

## 1. Install LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed]"
```

> Use LLaMA-Factory `>= 0.9.0`. MiniCPM 4 / 4.1 are text-only — no `[minicpm_v]` extra needed.

## 2. Prepare the dataset

LLaMA-Factory expects JSON in the **sharegpt** format. Save as `data/my_sft.json`:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Write a short article about edge AI."},
      {"from": "gpt", "value": "Edge AI is the practice of..."}
    ]
  },
  {
    "conversations": [
      {"from": "human", "value": "List three benefits of on-device inference."},
      {"from": "gpt", "value": "1. Lower latency...\n2. Better privacy...\n3. Offline availability..."}
    ]
  }
]
```

Register the dataset in `data/dataset_info.json`:

```json
{
  "my_sft": {
    "file_name": "my_sft.json",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations" }
  }
}
```

## 3. LoRA SFT

Create `configs/lora_minicpm4_1.yaml`:

```yaml
### model
model_name_or_path: openbmb/MiniCPM4.1-8B   # or openbmb/MiniCPM4-8B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: my_sft
template: minicpm4
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: saves/minicpm4_1-lora
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

Launch:

```bash
llamafactory-cli train configs/lora_minicpm4_1.yaml
```

> [!TIP]
> If the `minicpm4` template isn't recognised yet by your LLaMA-Factory version, fall back to `template: cpm` or rebuild from the latest `main` — MiniCPM 4 template support has been merged upstream.

## 4. Full fine-tuning

For full-parameter fine-tuning swap `finetuning_type: full` and bump GPU count via DeepSpeed:

```yaml
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json
```

Use 8 × A100 80G or equivalent. For most users LoRA + bf16 on 1–2 GPUs is enough.

## 5. Merge LoRA & export

```bash
llamafactory-cli export \
    --model_name_or_path openbmb/MiniCPM4.1-8B \
    --adapter_name_or_path saves/minicpm4_1-lora \
    --template minicpm4 \
    --finetuning_type lora \
    --export_dir saves/minicpm4_1-merged \
    --export_size 2 --export_legacy_format false
```

The exported directory is a standalone HF model — drop it into any of the [deployment guides](../deployment/) under MiniCPM 4 / 4.1.

## 6. Notes

- Dataset templates: SFT (`sharegpt`, `alpaca`), DPO (`dpo_pairs`), KTO (`kto_pairs`). Hybrid-reasoning training data should include `<think>...</think>` blocks in the assistant turn.
- Tokenizer is loaded from the original Hub model — no offline tokenizer surgery needed.
- For other fine-tuning toolkits (SWIFT, Align-anything, Transformers Trainer), see the shared finetune section in the sidebar.
