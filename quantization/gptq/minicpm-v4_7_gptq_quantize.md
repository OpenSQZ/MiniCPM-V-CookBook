# GPTQ

::::{Note}
**Support:** MiniCPM-V 4.7 (1B, 35B-A3B)
::::

> **1B only for a clean 4-bit checkpoint via AutoGPTQ.** MiniCPM-V 4.7 **35B-A3B** keeps its MoE
> experts as fused parameters, which AutoGPTQ cannot quantize (it only handles `nn.Linear`).
> For 35B-A3B use the **GGUF** variants or the published `-GPTQ` / `-INT4` / `-FP8` checkpoints
> (see the [FP8/INT4 (llm-compressor)](../llmcompressor/minicpm-v4_7_llmcompressor_quantize.md) guide).

> The AutoGPTQ fork used by MiniCPM-V is [`tc-mb/AutoGPTQ`](https://github.com/tc-mb/AutoGPTQ).
> The MiniCPM-V 4.7 1B text backbone is `qwen3_5_text`, so the existing
> `Qwen3_5GPTQForCausalLM` quantizer is reused directly.

## Method 1 — Use the pre-quantized model with vLLM

```python
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_NAME = "openbmb/MiniCPM-V-4.7-1B-GPTQ"

image = Image.open("image.png").convert("RGB")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

llm = LLM(model=MODEL_NAME, max_model_len=2048, trust_remote_code=True)

messages = [{"role": "user", "content": [
    {"type": "text", "text": "(<image>./</image>)\nPlease describe the content of this image"},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": image}},
    SamplingParams(temperature=0.0, max_tokens=1024),
)
print(outputs[0].outputs[0].text)
```

## Method 2 — Quantize the model yourself

### 1. Install AutoGPTQ

```bash
git clone https://github.com/tc-mb/AutoGPTQ.git
cd AutoGPTQ
pip install -e .
```

### 2. Quantization script

Extracts the Qwen3.5 text backbone from MiniCPM-V 4.7, quantizes it to 4-bit with GPTQ,
then reassembles the full model with the original multimodal components.

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

# ... reassemble the full state dict: prefix the quantized LLM tensors with
# "model.language_model." and merge in the non-LLM (vision) tensors, then save
# shards + index + config.json with quantization_config. See the full script.
```

A complete, ready-to-run version is in
[`minicpm-v4_7_gptq_quantize.py`](./minicpm-v4_7_gptq_quantize.py):

```bash
python minicpm-v4_7_gptq_quantize.py \
  --model-path openbmb/MiniCPM-V-4.7-1B \
  --save-path ./MiniCPM-V-4.7-1B-GPTQ
```

## Result (1B)

| Item | Value |
|---|---|
| Quant method | `gptq` |
| Bit width / group size | 4 / 128 |
| `desc_act` / sym | false / true |
| Size | ~1.9 GB (from 3.0 GB bf16) |

## Recommended sampling parameters

For instruct-mode generation we recommend: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.
