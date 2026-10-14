# AWQ

::::{Note}
**支持:** MiniCPM-V 4.7 (1B, 35B-A3B)
::::

> **只有 1B 能用 AutoAWQ 得到干净的 4-bit checkpoint。** MiniCPM-V 4.7 **35B-A3B** 的 MoE 专家以
> 融合参数存储，AutoAWQ 无法量化（它只处理 `nn.Linear`）。35B-A3B 请改用 **GGUF** 或已发布的
> `-AWQ` / `-INT4` / `-FP8` 权重（见 [FP8/INT4（llm-compressor）](../llmcompressor/minicpm-v4_7_llmcompressor_quantize_zh.md)）。

> MiniCPM-V 使用的 AutoAWQ 工作流托管在 [`tc-mb/AutoAWQ`](https://github.com/tc-mb/AutoAWQ)
> （上游 `casper-hansen/AutoAWQ` 已归档）。该 fork 增加了 `minicpmv4_7` 架构注册。

## 方法一 —— 直接用预量化模型 + vLLM

### 1. 下载预量化模型

```bash
git clone https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-AWQ
```

### 2. 用 vLLM 运行

```python
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_NAME = "openbmb/MiniCPM-V-4.7-1B-AWQ"
IMAGES = ["image.png"]

image = Image.open(IMAGES[0]).convert("RGB")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

llm = LLM(model=MODEL_NAME, max_model_len=8192, trust_remote_code=True)

messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "请描述这张图片的内容"},
    ],
}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": image}},
    SamplingParams(temperature=0.0, max_tokens=1024),
)
print(outputs[0].outputs[0].text)
```

## 方法二 —— 用 AutoAWQ 直接加载 AWQ 权重

### 1. 从源码安装 AutoAWQ

```bash
git clone https://github.com/tc-mb/AutoAWQ.git
cd AutoAWQ
pip install -e .
```

### 2. 推理脚本

```python
import torch
from PIL import Image
from transformers import AutoProcessor
from awq import AutoAWQForCausalLM

model_path = "openbmb/MiniCPM-V-4.7-1B-AWQ"
image_path = "./assets/airplane.jpeg"

model = AutoAWQForCausalLM.from_quantized(model_path, trust_remote_code=True).to("cuda")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

image = Image.open(image_path).convert("RGB")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "这张图片里有什么？"},
    ],
}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to("cuda")

with torch.inference_mode():
    out_ids = model.generate(**inputs, max_new_tokens=256)
print("输出:", processor.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## 方法三 —— 自己量化

### 1. 从源码安装 AutoAWQ

```bash
git clone https://github.com/tc-mb/AutoAWQ.git
cd AutoAWQ
pip install -e .
```

### 2. 量化脚本

```python
import os
import shutil
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

model_path = "openbmb/MiniCPM-V-4.7-1B"
quant_path = "./MiniCPM-V-4.7-1B-AWQ"

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
model.model.to("cuda:0")   # 确认骨干在 GPU 上
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def copy_files_not_in_B(A_path, B_path):
    if not os.path.exists(B_path):
        os.makedirs(B_path)
    files_in_A = {f for f in os.listdir(A_path) if not (".bin" in f or "safetensors" in f)}
    for f in files_in_A - set(os.listdir(B_path)):
        src, dst = os.path.join(A_path, f), os.path.join(B_path, f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)


def load_alpaca():
    data = load_dataset("tatsu-lab/alpaca", split="train")

    def concatenate(x):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": x["instruction"]},
            {"role": "assistant", "content": x["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)}

    return [r["text"] for r in data.map(concatenate)][:1024]


calib_data = load_alpaca()
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
copy_files_not_in_B(model_path, quant_path)
print(f'模型已量化并保存到 "{quant_path}"')
```

带命令行参数的即用版本见
[`minicpm-v4_7_awq_quantize.py`](./minicpm-v4_7_awq_quantize.py)：

```bash
python minicpm-v4_7_awq_quantize.py \
  --model-path openbmb/MiniCPM-V-4.7-1B \
  --save-path ./MiniCPM-V-4.7-1B-AWQ
```

## 结果（1B）

| 项 | 值 |
|---|---|
| 量化方法 | `awq` (GEMM) |
| 位宽 / group size | 4 / 128 |
| Zero point | 是 |
| 体积 | ~2.3 GB（bf16 为 3.0 GB） |

## 推荐采样参数

在 instruct 模式下生成时，推荐使用：`temperature=0.7`、`top_p=0.95`、`top_k=100`、`min_p=0.0`、`presence_penalty=0`、`repetition_penalty=1.03`。
