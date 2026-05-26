# GPTQ

:::{Note}
**Support:** MiniCPM-V 4.0
:::

## Method 1 (Use the pre-quantized model)

> [!NOTE]
> The HuggingFace mirror (`openbmb/MiniCPM-V-4-GPTQ`) hasn't been published yet — it will be online soon. For now download from **ModelScope** below, or use **Method 2** to quantize the model yourself.

### 1.Download the Model

* **ModelScope** — <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4-GPTQ>
* HuggingFace (coming soon) — <https://huggingface.co/openbmb/MiniCPM-V-4-GPTQ>

```Bash
# from ModelScope
git clone https://modelscope.cn/OpenBMB/MiniCPM-V-4-GPTQ
```

### 2.Run with vllm

```python
import os
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# Quantized model name or path
MODEL_NAME = "openbmb/MiniCPM-V-4-GPTQ"

# List of image file paths
IMAGES = [
    "image.png",
]

# Open and convert image
image = Image.open(IMAGES[0]).convert("RGB")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Initialize LLM
llm = LLM(
    model=MODEL_NAME, 
    # gpu_memory_utilization=0.9,
    max_model_len=2048,
    trust_remote_code=True,
    # disable_mm_preprocessor_cache=True,
    # limit_mm_per_prompt={"image": 5}
)

# Build messages
messages = [{
    "role": "user",
    "content": "(<image>./</image>)\nPlease describe the content of this image",
    # "content": "(<image>./</image>)\n请描述这张图片的内容",
}]

# Apply chat template to the messages
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Set stop token IDs
stop_tokens = ['<|im_end|>', '</s>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

# Set generation parameters
sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids,
    temperature=0.7,
    top_p=0.8,
    max_tokens=1024,
)

# Get model output
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
    }
}, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```


## Method 2 (Quantize the model yourself)

### 1.Download the Model
<!-- 下载模型
https://huggingface.co/openbmb/MiniCPM-V-4
 -->

Download the MiniCPM-V-4 model from [HuggingFace](https://huggingface.co/openbmb/MiniCPM-V-4)

```Bash
git clone https://huggingface.co/openbmb/MiniCPM-V-4
```

### 2.Install AutoGPTQ

```Bash
git clone https://github.com/tc-mb/AutoGPTQ.git
cd AutoGPTQ
pip install -e .
```

### 3.Quantization Script

The following script extracts the LLM backbone from MiniCPM-V-4, quantizes it to 4-bit using GPTQ, and reassembles the full model with the original multimodal components.

Run the following quantization script (replace `MODEL_PATH` and `OUTPUT_PATH` with the paths to the original model and the quantized model, respectively).

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
    """Pre-copy all .py files so transitive relative imports are available."""
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

MODEL_PATH = '/model/MiniCPM-V-4'
OUTPUT_PATH = './model/MiniCPM-V-4-gptq-int4'

BITS = 4
GROUP_SIZE = 128
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LEN = 512


def prepare_calibration_data(tokenizer, num_samples, max_length):
    """Prepare calibration data from Alpaca dataset."""
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
    llm.config.model_type = "llama"

    from auto_gptq import BaseQuantizeConfig
    from auto_gptq.modeling.llama import LlamaGPTQForCausalLM

    quantize_config = BaseQuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
    )
    logger.info(f"Quantize config: bits={BITS}, group_size={GROUP_SIZE}")

    llm.seqlen = MAX_SEQ_LEN
    gptq_model = LlamaGPTQForCausalLM(llm, False, quantize_config)

    logger.info("Starting GPTQ quantization...")
    gptq_model.quantize(examples, batch_size=1)
    logger.info("Quantization complete!")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    llm_temp_dir = os.path.join(OUTPUT_PATH, "_llm_temp")
    os.makedirs(llm_temp_dir, exist_ok=True)
    gptq_model.save_quantized(llm_temp_dir, use_safetensors=True)

    # Copy non-weight files from original model
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

    # Combine quantized LLM weights with original multimodal weights
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

    MAX_SHARD_SIZE_BYTES = 4 * 1024 * 1024 * 1024  # 4GB per shard
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

    # Save config with quantization info
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

