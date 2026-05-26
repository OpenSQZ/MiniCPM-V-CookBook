# LLaMA-Factory (MiniCPM-SALA)

> [!NOTE]
> MiniCPM-SALA fine-tuning uses LLaMA-Factory with the same flow as MiniCPM 4, but you need a **SALA-aware build of LLaMA-Factory** that recognises the hybrid sparse + linear attention layers. The main differences from the [MiniCPM 4 guide](llamafactory.html) are flagged below.

## 1. Install LLaMA-Factory + SALA dependencies

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed]"
```

Also install the SALA sparse-attention kernels (required for both training and inference on SALA):

```bash
git clone https://github.com/OpenBMB/infllmv2_cuda_impl.git
cd infllmv2_cuda_impl
pip install -e .
```

> [!IMPORTANT]
> The SALA modelling code is loaded from the Hub via `trust_remote_code`. If your LLaMA-Factory version doesn't recognise the `MiniCPMSALAForCausalLM` architecture, fall back to a plain Transformers Trainer recipe (see the [SALA technical report PDF](https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf) for the training hyperparameters used in the official release).

## 2. Prepare data

SALA's biggest selling point is long context, so prepare data that actually exercises it. Sample sharegpt entry (`data/long_sft.json`):

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Read this 50K-token document and answer the question that follows.\n\n<long document here>...\n\nQuestion: What is the main argument of section 3?"},
      {"from": "gpt", "value": "The main argument of section 3 is..."}
    ]
  }
]
```

Register in `data/dataset_info.json` as usual.

## 3. LoRA SFT config

`configs/lora_minicpm_sala.yaml`:

```yaml
### model
model_name_or_path: openbmb/MiniCPM-SALA
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: long_sft
template: minicpm4              # SALA inherits MiniCPM 4's chat template
cutoff_len: 65536               # raise this to exercise long-context layers
overwrite_cache: true

### output
output_dir: saves/sala-lora
save_steps: 100

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
num_train_epochs: 1.0
bf16: true

### sala-specific
flash_attn: fa2
use_unsloth: false
```

```bash
llamafactory-cli train configs/lora_minicpm_sala.yaml
```

## 4. Notes

- SALA is a **research checkpoint** — fine-tuning recipes may not be reproducible across releases. Pin the model and kernel commits if you need stability.
- The InfLLM-V2 sparse-attention kernel is required at training time. Without it, training falls back to a slow reference attention path that doesn't capture SALA's intended behaviour.
- For inference of the resulting adapter, use the [SALA SGLang guide](../deployment/sglang.html) with the merged checkpoint, or load it through HF Transformers directly.
