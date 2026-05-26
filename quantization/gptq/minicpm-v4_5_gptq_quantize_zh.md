# GPTQ

::::{Note}
**支持版本：** MiniCPM-V 4.5
::::

## 方法 1（用预量化模型）

### 1. 下载模型

从下面任一来源获取 4-bit GPTQ 量化的 MiniCPM-V-4_5：

* **HuggingFace** — <https://huggingface.co/openbmb/MiniCPM-V-4_5-GPTQ>
* **ModelScope** — <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-GPTQ>

```Bash
git clone https://huggingface.co/openbmb/MiniCPM-V-4_5-GPTQ
```

### 2. 用 vLLM 运行

```python
import os
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# 量化模型名称或本地路径
MODEL_NAME = "openbmb/MiniCPM-V-4_5-GPTQ"

# 图片文件列表
IMAGES = [
    "image.png",
]

# 加载并转换图片
image = Image.open(IMAGES[0]).convert("RGB")

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 初始化 LLM
llm = LLM(
    model=MODEL_NAME, 
    # gpu_memory_utilization=0.9,
    max_model_len=2048,
    trust_remote_code=True,
    # disable_mm_preprocessor_cache=True,
    # limit_mm_per_prompt={"image": 5}
)

# 构建消息
messages = [{
    "role": "user",
    "content": "(<image>./</image>)\n请描述这张图片的内容",
    # "content": "(<image>./</image>)\nPlease describe the content of this image",
}]

# 应用 chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 设置 stop token IDs
stop_tokens = ['<|im_end|>', '</s>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

# 采样参数
sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids,
    temperature=0.7,
    top_p=0.8,
    max_tokens=1024,
)

# 推理
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
    }
}, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```


## 方法 2（自己进行 GPTQ 量化）

### 1. 下载模型
<!-- 下载模型
https://huggingface.co/openbmb/MiniCPM-V-4_5
 -->

从 [HuggingFace](https://huggingface.co/openbmb/MiniCPM-V-4_5) 下载 MiniCPM-V 4.5 模型。

```Bash
git clone https://huggingface.co/openbmb/MiniCPM-V-4_5
```

### 2. 安装 AutoGPTQ

```Bash
git clone https://github.com/tc-mb/AutoGPTQ.git
cd AutoGPTQ
pip install -e .
```

### 3. 量化脚本

下方脚本会从 MiniCPM-V-4_5 中提取 LLM backbone（Qwen3），用 GPTQ 量化为 4-bit，再把量化后的权重与原始多模态部分重新组装。

执行下方脚本（按需替换 `MODEL_PATH` 与 `OUTPUT_PATH`）：

```python
import os
import sys
import json
import shutil
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig


def _ensure_hf_dynamic_module_cache(model_path: str) -> None:
    """提前复制所有 .py 文件，确保间接相对导入能解析。"""
    from transformers.dynamic_module_utils import (
        HF_MODULES_CACHE,
        TRANSFORMERS_DYNAMIC_MODULE_NAME,
        _sanitize_module_name,
        init_hf_modules,
    )
    init_hf_modules()
    submodule = _sanitize_module_name(Path(model_path).name)
    cache_dir = Path(HF_MODULES_CACHE) / TRANSFORMERS_DYNAMIC_MODULE_NAME / submodule
    cache_dir.mkdir(parents=True, exist_ok=True)
    for py in Path(model_path).glob("*.py"):
        dst = cache_dir / py.name
        if not dst.exists():
            shutil.copy2(str(py), str(dst))


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = '/model/MiniCPM-V-4_5'
OUTPUT_PATH = './model/MiniCPM-V-4_5-gptq-int4'

BITS = 4
GROUP_SIZE = 128
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LEN = 512


def prepare_calibration_data(tokenizer, num_samples, max_length):
    """从 Alpaca 数据集准备校准数据。"""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    examples = []
    for sample in dataset:
        text = sample.get("text", "") or sample.get("output", "") or sample.get("instruction", "")
        if not text:
            continue
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        examples.append({
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        })

    logger.info(f"Prepared {len(examples)} calibration samples")
    return examples


def main():
    logger.info(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    logger.info("Preparing calibration data...")
    examples = prepare_calibration_data(tokenizer, NUM_CALIBRATION_SAMPLES, MAX_SEQ_LEN)

    _ensure_hf_dynamic_module_cache(MODEL_PATH)
    logger.info(f"Loading full MiniCPM-V model from {MODEL_PATH}")
    full_model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    llm = full_model.llm
    logger.info(f"Extracted LLM backbone: {llm.__class__.__name__}")

    original_model_type = llm.config.model_type
    llm.config.model_type = "qwen3"

    from auto_gptq import BaseQuantizeConfig
    from auto_gptq.modeling.qwen3 import Qwen3GPTQForCausalLM

    quantize_config = BaseQuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
    )
    logger.info(f"Quantize config: bits={BITS}, group_size={GROUP_SIZE}")

    llm.seqlen = MAX_SEQ_LEN
    gptq_model = Qwen3GPTQForCausalLM(llm, False, quantize_config)

    logger.info("Starting GPTQ quantization...")
    gptq_model.quantize(examples, batch_size=1)
    logger.info("Quantization complete!")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    llm_temp_dir = os.path.join(OUTPUT_PATH, "_llm_temp")
    os.makedirs(llm_temp_dir, exist_ok=True)
    gptq_model.save_quantized(llm_temp_dir, use_safetensors=True)

    # 把原始模型中的非权重文件复制过来
    for fname in os.listdir(MODEL_PATH):
        src = os.path.join(MODEL_PATH, fname)
        dst = os.path.join(OUTPUT_PATH, fname)
        if fname.startswith("model") and fname.endswith(".safetensors"):
            continue
        if fname in ("config.json", "quantize_config.json", "model.safetensors.index.json"):
            continue
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
        elif os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)

    # 把量化后的 LLM 权重和原始多模态权重组合
    from safetensors.torch import load_file as safe_load, save_file as safe_save

    quantized_llm_files = [f for f in os.listdir(llm_temp_dir) if f.endswith(".safetensors")]
    llm_state_dict = {}
    for f in quantized_llm_files:
        sd = safe_load(os.path.join(llm_temp_dir, f))
        llm_state_dict.update(sd)

    non_llm_state_dict = {}
    full_model_sd = full_model.state_dict()
    for key, value in full_model_sd.items():
        if not key.startswith("llm."):
            non_llm_state_dict[key] = value

    combined_sd = {}
    for key, value in llm_state_dict.items():
        combined_sd[f"llm.{key}"] = value
    for key, value in non_llm_state_dict.items():
        combined_sd[key] = value

    MAX_SHARD_SIZE_BYTES = 4 * 1024 * 1024 * 1024  # 每分片 4GB
    sorted_keys = sorted(combined_sd.keys())

    shards = []
    current_shard = {}
    current_size = 0
    weight_map = {}

    for key in sorted_keys:
        tensor = combined_sd[key]
        tensor_size = tensor.nelement() * tensor.element_size()
        if current_size > 0 and current_size + tensor_size > MAX_SHARD_SIZE_BYTES:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor.clone().contiguous()
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    if len(shards) == 1:
        shard_name = "model.safetensors"
        safe_save(shards[0], os.path.join(OUTPUT_PATH, shard_name))
        for key in shards[0]:
            weight_map[key] = shard_name
    else:
        for i, shard in enumerate(shards):
            shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            safe_save(shard, os.path.join(OUTPUT_PATH, shard_name))
            for key in shard:
                weight_map[key] = shard_name

    index = {
        "metadata": {"total_size": sum(t.nelement() * t.element_size() for t in combined_sd.values())},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUTPUT_PATH, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # 保存带量化信息的 config
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config_dict = config.to_dict()
    config_dict["quantization_config"] = {
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "damp_percent": 0.01,
        "desc_act": False,
        "static_groups": False,
        "sym": True,
        "true_sequential": True,
        "quant_method": "gptq",
        "checkpoint_format": "gptq",
        "model_name_or_path": None,
        "model_file_base_name": None,
    }
    config_dict["model_type"] = original_model_type

    with open(os.path.join(OUTPUT_PATH, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    quant_config_dict = quantize_config.to_dict()
    with open(os.path.join(OUTPUT_PATH, "quantize_config.json"), "w") as f:
        json.dump(quant_config_dict, f, indent=2)

    shutil.rmtree(llm_temp_dir)

    del combined_sd, llm_state_dict, non_llm_state_dict, full_model_sd
    torch.cuda.empty_cache()

    logger.info(f"Done! Quantized model saved to: {OUTPUT_PATH}")
    logger.info(f"Quantization: W{BITS}A16 GPTQ (weight-only {BITS}-bit, activation fp16)")
    logger.info("Compatible with: transformers (via optimum) and vLLM")


if __name__ == "__main__":
    main()
```
