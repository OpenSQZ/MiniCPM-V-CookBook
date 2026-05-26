# Llama Factory

## Install LlamaFactory

1. Clone the LlamaFactory GitHub repository:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
```

2. Install LlamaFactory dependencies:

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,minicpm_v]"
```

## Prepare the Dataset

### Building Image Dataset

Refer to the **mllm_demo.json** dataset under [LLaMA-Factory/data](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json) and construct your data in the same format. The structure is as follows:

To use images in multi-turn conversations, add the `<image>` tag in the user's content for each turn, and add the corresponding image paths in the `images` field. The number of `<image>` tags should match the number of values in `images`.

```json
[
  {
    "messages": [
      {
        "content": "<image>Who are they?",
        "role": "user"
      },
      {
        "content": "They're Kane and Gretzka from Bayern Munich.",
        "role": "assistant"
      },
      {
        "content": "What are they doing?<image>",
        "role": "user"
      },
      {
        "content": "They are celebrating on the soccer field.",
        "role": "assistant"
      }
    ],
    "images": [
      "mllm_demo_data/1.jpg",
      "mllm_demo_data/1.jpg"
    ]
  },
  {
    "messages": [
      {
        "content": "<image>Who is he?",
        "role": "user"
      },
      {
        "content": "He's Thomas Muller from Bayern Munich.",
        "role": "assistant"
      },
      {
        "content": "Why is he on the ground?",
        "role": "user"
      },
      {
        "content": "Because he's sliding on his knees to celebrate.",
        "role": "assistant"
      }
    ],
    "images": [
      "mllm_demo_data/2.jpg"
    ]
  }
]
```

### Building Video Dataset

Refer to the **mllm_video_demo.json** dataset under [LLaMA-Factory/data](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json) and construct your data in the same format. The structure is as follows:

To use videos in multi-turn conversations, add the `<video>` tag in the user's content for each turn, and add the corresponding video paths in the `videos` field. The number of `<video>` tags should match the number of values in `videos`.

```json
[
  {
    "messages": [
      {
        "content": "<video>Why is this video funny?",
        "role": "user"
      },
      {
        "content": "Because a baby is reading, and he is so cute!",
        "role": "assistant"
      }
    ],
    "videos": [
      "mllm_demo_data/1.mp4"
    ]
  }
]
```

### Building Audio Dataset

**Note: Only MiniCPM-o 2.6 model supports audio fine-tuning**

Refer to the **mllm_audio_demo.json** dataset under [LLaMA-Factory/data](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json) and construct your data in the same format. The structure is as follows:

To use audio in multi-turn conversations, add the `<audio>` tag in the user's content for each turn, and add the corresponding audio paths in the `audios` field. The number of `<audio>` tags should match the number of values in `audios`.

```json
[
  {
    "messages": [
      {
        "content": "<audio>What's that sound?",
        "role": "user"
      },
      {
        "content": "It is the sound of glass shattering.",
        "role": "assistant"
      }
    ],
    "audios": [
      "mllm_demo_data/1.mp3"
    ]
  }
]
```

### Register Dataset

1. Name your constructed JSON file as `image_caption.json` and place it under `LLaMA-Factory/data/`.

2. Locate `LLaMA-Factory/data/dataset_info.json`.

   1. Search for `mllm_demo` and find the following field:

   ```json
      "mllm_demo": {
          "file_name": "mllm_demo.json",
          "formatting": "sharegpt",
          "columns": {
            "messages": "messages",
            "images": "images"
          }
    ```

   2. Change the **key** `mllm_demo` to your custom dataset name, e.g., `cpmv_img`.

   3. Change the `file_name` value to your constructed dataset name, e.g., `image_caption.json`.

   Example:

   ```json
   "cpmv_img": {
       "file_name": "image_caption.json",
       "formatting": "sharegpt",
       "columns": {
         "messages": "messages",
         "images": "images"
       },
       "tags": {
         "role_tag": "role",
         "content_tag": "content",
         "user_tag": "user",
         "assistant_tag": "assistant"
       }
   }
   ```

   4. For datasets containing videos and audio, please refer to the following format:

   ```json
   "mllm_video_audio_demo": {
     "file_name": "mllm_video_audio_demo.json",
     "formatting": "sharegpt",
     "columns": {
       "messages": "messages",
       "videos": "videos",
       "audios": "audios"
     },
     "tags": {
       "role_tag": "role",
       "content_tag": "content",
       "user_tag": "user",
       "assistant_tag": "assistant"
     }
   }
   ```

## Create Training Configuration YAML Files

### LoRA Fine-tuning

Create a configuration file named `minicpmv4_5_lora_sft.yaml` and place it in `LLaMA-Factory/minicpm_config`.

```yaml
### model
model_name_or_path: openbmb/MiniCPM-V-4_5 # Can be MiniCPM-V or MiniCPM-o local model
trust_remote_code: true

### method
stage: sft # sft training
do_train: true
finetuning_type: lora # LoRA fine-tuning
lora_target: q_proj,v_proj # LoRA layers to insert

### dataset
dataset: cpmv_img # Use the key you added in data/dataset_info.json
template: minicpm_v # Do not change
cutoff_len: 3072 # Model token length including multimodal
max_samples: 1000 # Max number of samples
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/minicpmv4_5/lora/sft
logging_steps: 1
save_steps: 100 # Save every N steps
plot_loss: true # Plot loss curve
overwrite_output_dir: true # Overwrite previous outputs
save_total_limit: 10

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 20.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
save_only_model: true

### eval
do_eval: false
```

### Full Fine-tuning

Create a full training configuration file `minicpmv4_5_full_sft.yaml` and place it in `LLaMA-Factory/minicpm_config`:

```yaml
### model
model_name_or_path: openbmb/MiniCPM-V-4_5 # MiniCPM-o-2_6 or MiniCPM-V-2_6 or local path
trust_remote_code: true
freeze_vision_tower: true # Freeze vision module
print_param_status: true
flash_attn: fa2 # Use flash attention 2

### method
stage: sft
do_train: true
finetuning_type: full # Full fine-tuning
deepspeed: configs/deepspeed/ds_z2_config.json # Use deepspeed zero2 distributed training
 
### dataset
dataset: cpmv_img # Use the key you added in data/dataset_info.json
template: minicpm_v
cutoff_len: 3072
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/minicpmv4_5/full/sft
logging_steps: 1
save_steps: 100
plot_loss: true
overwrite_output_dir: true
save_total_limit: 10

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 20.0
lr_scheduler_type: cosine
warmup_ratio: 0.1 # 10% warmup
bf16: true
ddp_timeout: 180000000
save_only_model: true

### eval
do_eval: false
```

## Model Training

### Full Training

```bash
cd LLaMA-Factory
llamafactory-cli train configs/minicpmv4_5_full_sft.yaml
```

### LoRA Training

1. Start training:

```bash
llamafactory-cli train configs/minicpmv4_5_lora_sft.yaml
```

2. Create a merge script `merge.yaml`:

```yaml
### model
model_name_or_path: openbmb/MiniCPM-V-4_5 # Original model path, can be local
adapter_name_or_path: saves/minicpm_v4_5/lora/sft # Path to saved LoRA model
template: minicpm_v
finetuning_type: lora
trust_remote_code: true

### export
export_dir: models/minicpmv4_5_lora_sft
export_size: 2
export_device: cpu
export_legacy_format: false
```

3. Merge the model:

```bash
llamafactory-cli export configs/minicpmv4_5_lora_export.yaml
```