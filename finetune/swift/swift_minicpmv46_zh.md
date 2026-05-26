# MiniCPM-V 4.6 微调教程 (ms-swift)

## 模型与任务概览

本章节以 counting 任务 [allenai/pixmo-count](https://huggingface.co/datasets/allenai/pixmo-count) 作为微调示例。

训练任务：

- 输入一张图片和一个计数问题
- `assistant` 目标输出需要先给出每个目标物体点的位置，用 `x y` 坐标表示，如 `<point>573 489</point>`，再输出最终数量，例如 `0`、`3`、`10`。


## 环境说明

**最小可运行安装步骤**

```bash
conda create -n "MiniCPM-V-4.6-Counting" python=3.10 -y
conda activate "MiniCPM-V-4.6-Counting"

pip install torch==2.8.0 torchvision==0.23.0

pip install \
  transformers==5.7.0 accelerate==1.13.0 \
  deepspeed==0.18.3 peft==0.18.1 trl==0.24.0 \
  wandb ninja einops safetensors tokenizers sentencepiece

MAX_JOBS=32 NVCC_THREADS=4 pip install --no-build-isolation flash-attn==2.8.3
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
pip install transformers==5.7.0
```

注意：MiniCPM-V 4.6 在 `transformers>=5.7.0` 版本被官方支持。请保证最后安装的 `transformers` 版本为 5.7.0 及以上。

**依赖版本参考**

```text
python                        3.10.0
accelerate                    1.13.0
deepspeed                     0.18.3
flash_attn                    2.8.3
ms_swift                      官方最新代码
torch                         2.8.0
torchvision                   0.23.0
transformers                  5.7.0
```

## 数据准备

从 [allenai/pixmo-count](https://huggingface.co/datasets/allenai/pixmo-count) 下载数据集，并将其转换为 ms-swift 格式。

**数据格式参考：**
  ```json
  {
      "messages": [
          {
              "content": "<image>\nCarefully observe the image. Are there any people in the image? If yes, please list their respective coordinates and provide the total count. If no, answer 0.",
              "role": "user"
          },
          {
              "content": "<think>\n\n</think>\n\nThe respective coordinates of people: <point>236 469</point>, <point>307 232</point>, <point>362 434</point>, <point>485 521</point>, <point>487 340</point>, <point>615 386</point>, <point>735 441</point>, <point>870 615</point>. So the total count is 8.",
              "role": "assistant"
          }
      ],
      "images": [
          "/path/to/images/*.jpg"
      ],
      "source_file": "pixmo-count",
      "orig_index": 1,
      "channel": "pixmo-count"
  }
  ```
- Counting 任务中加入对 points 预测的监督能够提高微调的效果，因此我们推荐将数据中的 points 坐标拼到 assistant 回复中。
- 由于 MiniCPM-V 4.6 会将图片坐标归一化到 `0~1000`，因此也需要对 points 坐标进行以下处理：
  ```python
  def expected_norm(x_px: float, y_px: float, width: int, height: int) -> Tuple[int, int]:
      return int((x_px / width) * 1000.0), int((y_px / height) * 1000.0)
  ```

## 启动训练

配置好模型路径、训练集路径、验证集路径和输出目录后，执行以下脚本即可以开始训练。

```bash
run_swift.sh
#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT="${MASTER_PORT:-29632}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-MiniCPMV46-Counting}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-mcpmv46_count}"
export WANDB_NAME="${WANDB_NAME:-mcpmv46_count}"

export DOWNSAMPLE_MODE="${DOWNSAMPLE_MODE:-4x}"

SWIFT_BIN="${SWIFT_BIN:-swift}"
MODEL_PATH="${MODEL_PATH:-/path/to/minicpm-v-4_6}"

TRAIN_DATA="${TRAIN_DATA:-/path/to/task_dataset/train/pixmo_count_train_with_channel}"
VALID_DATA="${VALID_DATA:-/path/to/task_dataset/val/validation-00000-of-00001-swift.parquet}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-zero2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-path/to/outdir}"

${SWIFT_BIN} sft \
  --model "${MODEL_PATH}" \
  --model_type minicpmv4_6 \
  --template minicpmv4_6 \
  --add_non_thinking_prefix true \
  --run_name "${WANDB_RUN_NAME}" \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VALID_DATA}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --freeze_vit False \
  --packing false \
  --max_length 4096 \
  --num_train_epochs 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-6 \
  --warmup_ratio 0.05 \
  --logging_steps 1 \
  --save_steps 132 \
  --eval_strategy steps \
  --eval_steps 80 \
  --save_total_limit 30 \
  --load_from_cache_file false \
  --dataset_num_proc 16 \
  --dataloader_num_workers 16 \
  --enable_channel_loss True \
  --attn_impl flash_attn \
  --output_dir "${OUTPUT_DIR}" \
  --report_to wandb
```

关键参数说明
- 训练支持 `16x`、`4x` 两种视觉 Token 压缩率，通过 `export DOWNSAMPLE_MODE="${DOWNSAMPLE_MODE:-4x}"` 参数进行控制。
- 当前版本的 `transformers` 对 Qwen3.5 系列的 packing 训练的支持仍存在问题，目前请使用 `--packing false`，官方修复后本文档也会进行更新。
- `--add_non_thinking_prefix true` 可以在训练提示词后加入 `<think>\n\n</think>\n\n`，实现无 thinking 训练。

## 训练过程

[https://wandb.ai/majy24-tsinghua-university/MiniCPMV46-Counting/reports/ms-swift---VmlldzoxNjgxMDk0Ng](https://wandb.ai/majy24-tsinghua-university/MiniCPMV46-Counting/reports/ms-swift---VmlldzoxNjgxMDk0Ng)

<img src="../assets/finetune_minicpmv46/minicpmv46_swift_ft_dynamics.png" alt="ms-swift 训练过程" />



## 评测结果

评测指标说明：

| 指标 | 说明 |
| --- | --- |
| Acc@0 | 精确匹配率，即预测值与真实值完全一致 |
| Acc@0 Top1 | 训练过程保存的所有 checkpoint 中，评测结果 Acc@0 的最高分数 |
| Acc@0 Avg.Top3 | 训练过程保存的所有 checkpoint 中，评测结果 Acc@0 前三名的平均分数 |

下表展示了两种视觉 Token 压缩率设置下的评测结果：

| 模型               | 视觉 Token 压缩率 | Acc@0 Top1 | Acc@0 Avg.Top3 |
| ---------------- | ------------ | ---------- | -------------- |
| MiniCPM-V 4.6    | 16           | 46.5       | N/A  [1]     |
| MiniCPM-V 4.6    | 4            | 51.8       | N/A  [1]      |
| Fine-tuned model | 16           | 79.7       | 79.3           |
| Fine-tuned model | 4            | 84.3       | 83.9           |

<small>[1]: MiniCPM-V 4.6 为原始模型，未经微调，仅有一个 Acc@0 结果 (Acc@0 Top1)，无法计算 Acc@0 Avg.Top3</small>

输出样例：
  ```text
  Q: Carefully observe the image. Are there any airplanes in the image? If yes, please list their respective coordinates and provide the total count. If no, answer 0.

  A: The respective coordinates of airplanes: <point>310 370</point>, <point>365 277</point>, <point>388 486</point>, <point>405 185</point>, <point>437 368</point>, <point>474 611</point>, <point>503 250</point>, <point>527 451</point>, <point>535 818</point>, <point>597 331</point>. So the total count is 10.
  ```
  <img src="../assets/finetune_minicpmv46/sample_1.png" alt="ms-swift sample" />
