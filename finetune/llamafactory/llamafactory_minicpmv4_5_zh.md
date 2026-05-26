# Llama Factory

## 安装LlamaFactory

1. 获取LlamaFactory Github代码

```Python
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
```

1. 安装LlamaFactory的依赖

```Python
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,minicpm_v]
```

## 构造数据

### 构建图片数据集
参照LLaMA-Factory/[data](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json)下的**mllm_demo.json**数据集,按照相同格式构造数据，结构如下：

如需在多轮对话中使用图片，请在每轮对话的user content中添加`<image>`标签，并在images中添加相应的图片路径。`<image>` 标签数量需要与 `images`中的值数量相匹配。

```JSON
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

### 构建视频数据集

参照LLaMA-Factory/[data](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json)下的**mllm_video_demo.json**数据集,按照相同格式构造数据，结构如下：

如需在多轮对话中使用图片，请在每轮对话的user content中添加`<video>`标签，并在images中添加相应的图片路径。`<video>` 标签数量需要与 `videos`中的值数量相匹配。

```JSON
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

### 构建音频数据集

**注意：仅MiniCPM-o 2.6模型支持音频微调**

参照LLaMA-Factory/[data](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json)下的**mllm_audio_demo.json**数据集,按照相同格式构造数据，结构如下：

如需在多轮对话中使用图片，请在每轮对话的user content中添加`<audio>`标签，并在images中添加相应的图片路径。`<audio>` 标签数量需要与 `audios`中的值数量相匹配。

```JSON
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

### 注册数据集
1. 将构造的JSON文件命名为：image_caption.json，并放到LLaMA-Factory/[data/](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json)路径下

2. 找到 LLaMA-Factory/[data/dataset_info.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json)

   1. 搜索`mllm_demo`,找到以下字段

    ```JSON
      "mllm_demo": {
          "file_name": "mllm_demo.json",
          "formatting": "sharegpt",
          "columns": {
            "messages": "messages",
            "images": "images"
          }
    ```

   2.  将**键值**`mllm_demo`改成自定义的数据集名称，如cpmv_img

   3.  将`file_name`应的值改成构造的数据集名称，如上文的image_caption.json

   Example:

    ```JSON
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
    4.  对于包含视频和音频的数据集，请参照下列格式

    ```JSON
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

## 创建训练的配置yaml文件：

### Lora微调

创建minicpmv4_5_lora_sft.yaml的配置文件，并且放入LLaMA-Factory/minicpm_config。

```YAML
### model
model_name_or_path: openbmb/MiniCPM-V-4_5 # 可以是MiniCPM-V或者MiniCPM-o的本地模型
trust_remote_code: true

### method
stage: sft # sft训练
do_train: true
finetuning_type: lora # lora微调
lora_target: q_proj,v_proj # lora层插入哪里

### dataset
dataset: cpmv_img # 改成你上面data/data_info.json的文件下新增的键名
template: minicpm_v # 不要改
cutoff_len: 3072 # 包多模态在哪占用的模型token长度
max_samples: 1000 #最多用多少条数据
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/minicpmv4_5/lora/sft
logging_steps: 1
save_steps: 100 #多少步报错一次
plot_loss: true # 是否绘制损失函数
overwrite_output_dir: true #是否覆盖之前的保存
save_total_limit: 10

### train
per_device_train_batch_size: 2 #训练batch_size
gradient_accumulation_steps: 1 #梯度累积次数
learning_rate: 1.0e-5 # 学习率
num_train_epochs: 20.0 #最多训练轮次
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
save_only_model: true

### eval
do_eval: false
```

### 全量微调

创建全量训练配置minicpmv4_5_full_sft.yaml文件，并且放入LLaMA-Factory/minicpm_config：

```YAML
### model
model_name_or_path: openbmb/MiniCPM-V-4_5 # 可以是MiniCPM-V或者MiniCPM-o的本地模型
trust_remote_code: true
freeze_vision_tower: true #冻结图像模块
print_param_status: true
flash_attn: fa2 #使用flash attn2

### method
stage: sft
do_train: true
finetuning_type: full #全量微调
deepspeed: configs/deepspeed/ds_z2_config.json # deepspeed使用zero2分布式训练
 
### dataset
dataset: cpmv_img # 改成你上面data/data_info.json的文件下新增的键名
template: minicpm_v #
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
warmup_ratio: 0.1 # warmup(学习率上升)占训练数据的10%
bf16: true #bf16精度
ddp_timeout: 180000000
save_only_model: true

### eval
do_eval: false
```

## 模型训练

### 全量训练

```Bash
cd LLaMA-Factory
llamafactory-cli train configs/minicpmv4_5_full_sft.yaml
```

### Lora训练

1. 开始训练

```Bash
llamafactory-cli train configs/minicpmv4_5_lora_sft.yaml
```

2. 创建合并脚本merge.yaml

```Bash
### model
model_name_or_path: openbmb/MiniCPM-V-4_5 # 这里可以填入原始模型地址，可以是本地模型
adapter_name_or_path: saves/minicpm_v4_5/lora/sft # 这里填入保存的lora模型地址
template: minicpm_v
finetuning_type: lora
trust_remote_code: true

### export
export_dir: models/minicpmv4_5_lora_sft
export_size: 2
export_device: cpu
export_legacy_format: false
```

3. 合并模型

```Bash
llamafactory-cli export configs/minicpmv4_5_lora_export.yaml
```