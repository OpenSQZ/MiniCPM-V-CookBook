# MiniCPM-V 4.7 35B-A3B Fine-Tuning Tutorial (ms-swift)

## Model and Task Overview

This section uses the counting task [allenai/pixmo-count](https://huggingface.co/datasets/allenai/pixmo-count) as a fine-tuning example.

Training task:

- Input an image and a counting question
- The target `assistant` response outputs the final count as a single integer, such as `0`, `3`, or `10`.

## Environment Setup

**Minimum runnable installation steps**

```bash
conda create -n "MiniCPM-V-4.7-35B-A3B-Counting" python=3.12 -y
conda activate "MiniCPM-V-4.7-35B-A3B-Counting"

pip install torch==2.10.0 torchvision==0.25.0

pip install \
  transformers==5.18.0 accelerate==1.15.0 \
  deepspeed==0.18.3 peft==0.18.1 trl==0.24.0 \
  flash-linear-attention==0.5.2 wandb ninja einops safetensors tokenizers sentencepiece

MAX_JOBS=32 NVCC_THREADS=4 pip install --no-build-isolation --no-binary causal-conv1d causal-conv1d==1.7.0
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git fetch origin pull/10338/head
git checkout FETCH_HEAD
pip install -e .
pip install transformers==5.18.0
```

Note: MiniCPM-V 4.7 35B-A3B is officially supported by `transformers>=5.18.0`. Please make sure the final version of `transformers` in your environment is larger or equal to `5.18.0`.

**Reference dependency versions**

```text
python                        3.12.3
accelerate                    1.15.0
deepspeed                     0.18.3
flash-linear-attention        0.5.2
ms_swift                      4.6.0.dev0 (PR #10338)
torch                         2.10.0+cu128
torchvision                   0.25.0
transformers                  5.18.0
```

### 2.2 Data Preparation

Download the dataset from [allenai/pixmo-count](https://huggingface.co/datasets/allenai/pixmo-count) and convert it into the ms-swift format.

**Data format reference:**
  ```json
  {
      "messages": [
          {
              "content": "<image>\nCarefully observe the image. How many people are there in the image? Answer with a single integer only.",
              "role": "user"
          },
          {
              "content": "8",
              "role": "assistant"
          }
      ],
      "images": [
          "/path/to/images/*.jpg"
      ],
      "source_file": "pixmo-count-train",
      "orig_index": 1,
      "channel": "pixmo-count-no-points"
  }
  ```

## Launch Training

After configuring the model path, training set path, validation set path, and output directory, run the following script to start training.

```bash
# run_swift.sh
#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29632}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-MiniCPMV47-35B-A3B-Counting}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-mcpmv47_35b_count}"
export WANDB_NAME="${WANDB_NAME:-mcpmv47_35b_count}"

export DOWNSAMPLE_MODE="${DOWNSAMPLE_MODE:-4x}"

SWIFT_BIN="${SWIFT_BIN:-swift}"
MODEL_PATH="${MODEL_PATH:-/path/to/MiniCPM-V-4.7-35B-A3B}"

TRAIN_DATA="${TRAIN_DATA:-/path/to/task_dataset/train.jsonl}"
VALID_DATA="${VALID_DATA:-/path/to/task_dataset/validation.jsonl}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero3}"
if [[ "${DEEPSPEED_CONFIG}" == zero3 ]]; then
    DEEPSPEED_CONFIG="$(python - <<'PY'
import json
from importlib.resources import files

config = json.loads(files('swift').joinpath('config/zero3.json').read_text())
config['zero_optimization']['sub_group_size'] = 100000000
print(json.dumps(config))
PY
)"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-path/to/outdir}"

${SWIFT_BIN} sft \
  --model "${MODEL_PATH}" \
  --model_type minicpmv4_7 \
  --template minicpmv4_7 \
  --add_non_thinking_prefix true \
  --run_name "${WANDB_RUN_NAME}" \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VALID_DATA}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --experts_impl grouped_mm \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --freeze_vit False \
  --freeze_aligner False \
  --packing false \
  --padding_side left \
  --gradient_checkpointing true \
  --max_length 4096 \
  --num_train_epochs 4 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
  --learning_rate 5e-6 \
  --warmup_ratio 0.05 \
  --logging_steps 1 \
  --save_steps 132 \
  --eval_strategy steps \
  --eval_steps 80 \
  --save_total_limit 30 \
  --load_from_cache_file false \
  --dataset_num_proc 8 \
  --dataloader_num_workers 4 \
  --enable_channel_loss True \
  --attn_impl sdpa \
  --loss_scale ignore_empty_think \
  --output_dir "${OUTPUT_DIR}" \
  --report_to wandb
```

Key parameter notes

- Training supports two visual token compression ratios, `16x` and `4x`, which are controlled by `export DOWNSAMPLE_MODE="${DOWNSAMPLE_MODE:-4x}"`.
- For this Qwen3.5-based model, please use `--packing false`, as in the verified training runs.
- `--add_non_thinking_prefix true` can append `<think>\n\n</think>\n\n` to training prompts, achieve no-thinking training.

## Training Curves

<img src="../assets/finetune_minicpmv47_35b/minicpmv47_35b_swift_ft_dynamics.png" alt="ms-swift training curves" />

## Evaluation Results

Evaluation metrics explanation:

| Metric | Description |
| --- | --- |
| Acc@0 | Exact match accuracy (predicted value = ground truth) |
| Acc@0 Top1 | The test Acc@0 score of the checkpoint with the highest validation Acc@0 |
| Acc@0 Avg.Top3 | The average test Acc@0 score of the top three checkpoints ranked by validation Acc@0 |

The table below shows the results under two visual token compression ratios:

| Model | Visual Token Compression Ratio | Acc@0 Top1 | Acc@0 Avg.Top3 |
| --- | --- | --- | --- |
| MiniCPM-V 4.7 35B-A3B | 16 | 69.1 | N/A [1] |
| MiniCPM-V 4.7 35B-A3B | 4 | 73.2 | N/A [1] |
| Fine-tuned model | 16 | 81.4 | 81.3 |
| Fine-tuned model | 4 | 85.0 | 84.3 |

  <small>[1]: MiniCPM-V 4.7 35B-A3B is the original model without fine-tuning, so only one Acc@0 result (Acc@0 Top1) is available and Acc@0 Avg.Top3 cannot be computed.</small>

Output example:

  ```text
  Q: Carefully observe the image. How many airplanes are there in the image? Answer with a single integer only.

  A: 7
  ```

  <img src="../assets/finetune_minicpmv47_35b/minicpmv47_35b_sample_prediction.png" alt="ms-swift sample" />

## Recommended sampling parameters

For instruct-mode generation we recommend: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.
