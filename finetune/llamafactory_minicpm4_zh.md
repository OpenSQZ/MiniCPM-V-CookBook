# LLaMA-Factory（MiniCPM 4 / 4.1）

> [!NOTE]
> [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 是一站式的 SFT / DPO / RLHF 微调框架。MiniCPM 4 / 4.1 走 LLaMA-Factory 上游主线，无需 fork。

## 1. 安装 LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed]"
```

> 使用 LLaMA-Factory `>= 0.9.0`。MiniCPM 4 / 4.1 是纯文本模型，无需 `[minicpm_v]` 扩展。

## 2. 准备数据

LLaMA-Factory 使用 **sharegpt** 格式 JSON。保存为 `data/my_sft.json`：

```json
[
  {
    "conversations": [
      {"from": "human", "value": "写一篇关于端侧 AI 的短文。"},
      {"from": "gpt", "value": "端侧 AI 指的是..."}
    ]
  },
  {
    "conversations": [
      {"from": "human", "value": "列举三个端侧推理的好处。"},
      {"from": "gpt", "value": "1. 延迟更低...\n2. 隐私更好...\n3. 可离线..."}
    ]
  }
]
```

在 `data/dataset_info.json` 中注册数据集：

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

新建 `configs/lora_minicpm4_1.yaml`：

```yaml
### 模型
model_name_or_path: openbmb/MiniCPM4.1-8B   # 或 openbmb/MiniCPM4-8B
trust_remote_code: true

### 方法
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### 数据
dataset: my_sft
template: minicpm4
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 8

### 输出
output_dir: saves/minicpm4_1-lora
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### 训练
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

启动：

```bash
llamafactory-cli train configs/lora_minicpm4_1.yaml
```

> [!TIP]
> 如果当前 LLaMA-Factory 版本未识别 `minicpm4` 模板，可临时使用 `template: cpm`，或从最新 `main` 重新安装 —— MiniCPM 4 模板已合入上游。

## 4. 全参微调

全参微调把 `finetuning_type` 改为 `full`，配合 DeepSpeed 在多卡上训练：

```yaml
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json
```

建议 8 × A100 80G 或同等配置。多数场景 1–2 卡 LoRA + bf16 已经够用。

## 5. 合并 LoRA 与导出

```bash
llamafactory-cli export \
    --model_name_or_path openbmb/MiniCPM4.1-8B \
    --adapter_name_or_path saves/minicpm4_1-lora \
    --template minicpm4 \
    --finetuning_type lora \
    --export_dir saves/minicpm4_1-merged \
    --export_size 2 --export_legacy_format false
```

导出的目录就是独立 HF 模型，可直接接入 MiniCPM 4 / 4.1 下的任意[部署指南](../deployment/)。

## 6. 注意事项

- 数据模板：SFT（`sharegpt`、`alpaca`）、DPO（`dpo_pairs`）、KTO（`kto_pairs`）。训练混合思考数据时，assistant 段需要包含 `<think>...</think>` 块。
- Tokenizer 直接从 Hub 原始模型加载，无需离线 tokenizer 手术。
- 其他微调工具（SWIFT、Align-anything、Transformers Trainer）参考侧边栏的 shared finetune 章节。
