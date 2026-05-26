# SWIFT

:::{Note}
**Support:** MiniCPM-V 2.6
:::

SWIFT is an efficient and scalable framework for fine-tuning large pre-trained models with support for various parameter-efficient methods like LoRA, Adapter, and Prompt Tuning.

## SWIFT Install

You can quickly install SWIFT using bash commands.

``` bash
git clone https://github.com/modelscope/swift.git
cd swift
pip install -r requirements.txt
pip install -e '.[llm]'
```

## Train

### Prepare the data

You can refer to the format below to construct your dataset. Custom datasets support JSON and JSONL formats.

``` json
{"query": "What does this picture describe?", "response": "This picture has a giant panda.", "images": ["local_image_path"]}
{"query": "What does this picture describe?", "response": "This picture has a giant panda.", "history": [], "images": ["local_image_path"]}
{"query": "Is bamboo tasty?", "response": "It seems pretty tasty judging by the panda's expression.", "history": [["What's in this picture?", "There's a giant panda in this picture."], ["What is the panda doing?", "Eating bamboo."]], "images": ["image_url"]}
```

Alternatively, you can also use datasets from ModelScope, such as the image dataset [coco-en-mini](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary) or the video dataset [video-chatgpt](https://modelscope.cn/datasets/swift/VideoChatGPT).

### Image Fine-tuning

We use the `coco-en-mini` dataset for fine-tuning, which involves describing the content of images. 

The following is the code configuration:

``` bash
# By default, `lora_target_modules` will be set to all linear layers in `llm` and `resampler`
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
  --model_type minicpm-v-v2_6-chat \
  --model_id_or_path OpenBMB/MiniCPM-V-2_6 \
  --sft_type lora \
  --dataset coco-en-mini#20000 \
  --deepspeed default-zero2
```
If you want to use a custom dataset, simply specify it as follows:

``` bash
  --dataset train.jsonl \
  --val_dataset val.jsonl \
```

The inference script after fine-tuning is as follows:

```bash
# Set `--show_dataset_sample -1` to run full evaluation
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/minicpm-v-v2_6-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true --merge_lora true
```

### Video Fine-tuning

We use the `video-chatgpt` dataset for fine-tuning, which involves describing the content of images. 

The following is the code configuration:

``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
  --model_type minicpm-v-v2_6-chat \
  --model_id_or_path OpenBMB/MiniCPM-V-2_6 \
  --sft_type lora \
  --dataset video-chatgpt \
  --deepspeed default-zero2
```
If you want to use a custom dataset, simply specify it as follows:

``` bash
  --dataset train.jsonl \
  --val_dataset val.jsonl \
```

Custom datasets support JSON and JSONL formats. Below is an example of the video dataset:

```json
{"query": "<video>Describe what is happening in this video.", "response": "A dog is playing with a ball in a park.", "videos": ["path/to/video1.mp4"]}
{"query": "What are the people doing in the video?<video>Can you see any vehicles?<video>", "response": "People are walking on the street, and there are cars and bicycles.", "history": [], "videos": ["path/to/video2.mp4", "path/to/video3.mp4"]}
{"query": "Was there a red car in the previous video?", "response": "Yes, there was a red car parked near the sidewalk.", "history": [["What did you see in the video?", "There was a car, a bicycle, and several pedestrians."], ["What time was it?", "It seemed to be in the afternoon."]], "videos": []}
```

The inference script after fine-tuning is as follows:

```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/minicpm-v-v2_6-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true --merge_lora true
```

## Infer

Run the bash code will download the model of MiniCPM-V 2.6 and run the inference

```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
  --model_type minicpm-v-v2_6-chat \
  --model_id_or_path OpenBMB/MiniCPM-V-2_6
```
