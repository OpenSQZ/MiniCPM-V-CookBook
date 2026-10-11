# GPTQ

::::{Note}
**支持:** MiniCPM-V 4.7 (1B, 35B-A3B)
::::

> **只有 1B 能用 AutoGPTQ 得到干净的 4-bit checkpoint。** MiniCPM-V 4.7 **35B-A3B** 的 MoE 专家以
> 融合参数存储，AutoGPTQ 无法量化（它只处理 `nn.Linear`）。35B-A3B 请改用 **GGUF** 或已发布的
> `-GPTQ` / `-INT4` / `-FP8` 权重（见 [FP8/INT4（llm-compressor）](../llmcompressor/minicpm-v4_7_llmcompressor_quantize_zh.md)）。

> MiniCPM-V 使用的 AutoGPTQ fork 为 [`tc-mb/AutoGPTQ`](https://github.com/tc-mb/AutoGPTQ)。
> MiniCPM-V 4.7 1B 的文本骨干是 `qwen3_5_text`，因此直接复用既有的
> `Qwen3_5GPTQForCausalLM` 量化器。

## 方法一 —— 直接用预量化模型 + vLLM

```python
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_NAME = "openbmb/MiniCPM-V-4.7-1B-GPTQ"

image = Image.open("image.png").convert("RGB")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

llm = LLM(model=MODEL_NAME, max_model_len=2048, trust_remote_code=True)

messages = [{"role": "user", "content": [
    {"type": "text", "text": "(<image>./</image>)\n请描述这张图片的内容"},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": image}},
    SamplingParams(temperature=0.0, max_tokens=1024),
)
print(outputs[0].outputs[0].text)
```

## 方法二 —— 自己量化

### 1. 安装 AutoGPTQ

```bash
git clone https://github.com/tc-mb/AutoGPTQ.git
cd AutoGPTQ
pip install -e .
```

### 2. 量化脚本

从 MiniCPM-V 4.7 中抽出 Qwen3.5 文本骨干，用 GPTQ 量化为 4-bit，再用原始多模态组件
重组完整模型。

```python
import os, json, shutil, logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForImageTextToText
from safetensors.torch import load_file as safe_load, save_file as safe_save

MODEL_PATH = "openbmb/MiniCPM-V-4.7-1B"
OUTPUT_PATH = "./MiniCPM-V-4.7-1B-GPTQ"
BITS, GROUP_SIZE = 4, 128
NUM_CALIBRATION_SAMPLES, MAX_SEQ_LEN = 128, 256


def prepare_calibration_data(tokenizer, num_samples, max_length):
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    examples = []
    for sample in dataset:
        text = sample.get("text", "") or sample.get("output", "") or sample.get("instruction", "")
        if not text:
            continue
        tok = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        examples.append({"input_ids": tok["input_ids"].squeeze(0),
                         "attention_mask": tok["attention_mask"].squeeze(0)})
    return examples


class LLMWrapper(torch.nn.Module):
    def __init__(self, text_model, lm_head, config):
        super().__init__()
        self.model, self.lm_head, self.config = text_model, lm_head, config
        self.seqlen = MAX_SEQ_LEN

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[0]
        return self.lm_head(hidden)


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
examples = prepare_calibration_data(tokenizer, NUM_CALIBRATION_SAMPLES, MAX_SEQ_LEN)
full_model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=torch.float16,
                                                         device_map="cpu", trust_remote_code=True)
text_model, lm_head = full_model.model.language_model, full_model.lm_head
config = full_model.config.text_config
config.model_type = "qwen3_5"

from auto_gptq import BaseQuantizeConfig
from auto_gptq.modeling.qwen3_5 import Qwen3_5GPTQForCausalLM

gptq_model = Qwen3_5GPTQForCausalLM(
    LLMWrapper(text_model, lm_head, config), False,
    BaseQuantizeConfig(bits=BITS, group_size=GROUP_SIZE, desc_act=False, sym=True),
)
gptq_model.quantize(examples, batch_size=1)

# ... 重组完整 state dict：把量化后的 LLM 张量加前缀 "model.language_model."，
# 再合并非 LLM（视觉）张量，保存分片 + index + 带 quantization_config 的 config.json。
# 完整脚本见下方 .py。
```

完整可运行版本见
[`minicpm-v4_7_gptq_quantize.py`](./minicpm-v4_7_gptq_quantize.py)：

```bash
python minicpm-v4_7_gptq_quantize.py \
  --model-path openbmb/MiniCPM-V-4.7-1B \
  --save-path ./MiniCPM-V-4.7-1B-GPTQ
```

## 结果（1B）

| 项 | 值 |
|---|---|
| 量化方法 | `gptq` |
| 位宽 / group size | 4 / 128 |
| `desc_act` / sym | false / true |
| 体积 | ~1.9 GB（bf16 为 3.0 GB） |
