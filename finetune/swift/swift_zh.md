# SWIFT

:::{Note}
**支持版本：** MiniCPM-V 2.6
:::

SWIFT 是一个高效、可扩展的大模型微调框架，支持 LoRA、Adapter、Prompt Tuning 等多种参数高效微调方法。

## 安装 SWIFT

可以用以下命令快速安装 SWIFT：

``` bash
git clone https://github.com/modelscope/swift.git
cd swift
pip install -r requirements.txt
pip install -e '.[llm]'
```

## 训练

### 准备数据

可以参考下方格式准备自己的数据集。自定义数据集支持 JSON 与 JSONL 格式。

``` json
{"query": "What does this picture describe?", "response": "This picture has a giant panda.", "images": ["local_image_path"]}
{"query": "What does this picture describe?", "response": "This picture has a giant panda.", "history": [], "images": ["local_image_path"]}
{"query": "Is bamboo tasty?", "response": "It seems pretty tasty judging by the panda's expression.", "history": [["What's in this picture?", "There's a giant panda in this picture."], ["What is the panda doing?", "Eating bamboo."]], "images": ["image_url"]}
```

也可以直接使用 ModelScope 上的数据集，例如图像数据集 [coco-en-mini](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary) 或视频数据集 [video-chatgpt](https://modelscope.cn/datasets/swift/VideoChatGPT)。

### 图像微调

我们使用 `coco-en-mini` 数据集做微调，任务是描述图片内容。

下面是脚本配置：

``` bash
# 默认情况下，`lora_target_modules` 会被设置成 `llm` 与 `resampler` 中所有的 linear 层
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
  --model_type minicpm-v-v2_6-chat \
  --model_id_or_path OpenBMB/MiniCPM-V-2_6 \
  --sft_type lora \
  --dataset coco-en-mini#20000 \
  --deepspeed default-zero2
```

如果想用自定义数据集，按下面方式指定即可：

``` bash
  --dataset train.jsonl \
  --val_dataset val.jsonl \
```

微调后的推理脚本如下：

```bash
# 设 `--show_dataset_sample -1` 跑完整评测
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/minicpm-v-v2_6-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true --merge_lora true
```

### 视频微调

我们使用 `video-chatgpt` 数据集做微调，任务是描述视频内容。

下面是脚本配置：

``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
  --model_type minicpm-v-v2_6-chat \
  --model_id_or_path OpenBMB/MiniCPM-V-2_6 \
  --sft_type lora \
  --dataset video-chatgpt \
  --deepspeed default-zero2
```

如果想用自定义数据集：

``` bash
  --dataset train.jsonl \
  --val_dataset val.jsonl \
```

自定义数据集支持 JSON 与 JSONL 格式。下面是视频数据集示例：

```json
{"query": "<video>Describe what is happening in this video.", "response": "A dog is playing with a ball in a park.", "videos": ["path/to/video1.mp4"]}
{"query": "What are the people doing in the video?<video>Can you see any vehicles?<video>", "response": "People are walking on the street, and there are cars and bicycles.", "history": [], "videos": ["path/to/video2.mp4", "path/to/video3.mp4"]}
{"query": "Was there a red car in the previous video?", "response": "Yes, there was a red car parked near the sidewalk.", "history": [["What did you see in the video?", "There was a car, a bicycle, and several pedestrians."], ["What time was it?", "It seemed to be in the afternoon."]], "videos": []}
```

微调后的推理脚本如下：

```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/minicpm-v-v2_6-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true --merge_lora true
```

## 推理

下面这段命令会下载 MiniCPM-V 2.6 模型并直接推理：

```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
  --model_type minicpm-v-v2_6-chat \
  --model_id_or_path OpenBMB/MiniCPM-V-2_6
```
