"""
Quantize MiniCPM-V 4.6's LLM backbone (Qwen3_5TextModel) to 4-bit using GPTQ.
Output is compatible with both transformers and vLLM for inference.

Download the model from https://huggingface.co/openbmb/MiniCPM-V-4_6
Install AutoGPTQ from source:
  git clone https://github.com/tc-mb/AutoGPTQ.git && cd AutoGPTQ && pip install -e .
Then run:
  python quantization/gptq/minicpm-v4_6_gptq_quantize.py
"""
import os
import json
import shutil
import logging

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForImageTextToText
from safetensors.torch import load_file as safe_load, save_file as safe_save

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/model/MiniCPM-V-4_6"
OUTPUT_PATH = "./model/MiniCPM-V-4_6-gptq-int4"

BITS = 4
GROUP_SIZE = 128
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LEN = 256


def prepare_calibration_data(tokenizer, num_samples, max_length):
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


class LLMWrapper(torch.nn.Module):
    """Wraps Qwen3_5TextModel + lm_head to look like a CausalLM for GPTQ."""
    def __init__(self, text_model, lm_head, config):
        super().__init__()
        self.model = text_model
        self.lm_head = lm_head
        self.config = config
        self.seqlen = MAX_SEQ_LEN

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs[0]
        logits = self.lm_head(hidden)
        return logits


def main():
    logger.info(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    logger.info("Preparing calibration data...")
    examples = prepare_calibration_data(tokenizer, NUM_CALIBRATION_SAMPLES, MAX_SEQ_LEN)

    logger.info(f"Loading full model from {MODEL_PATH}")
    full_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="cpu",
    )

    text_model = full_model.model.language_model
    lm_head = full_model.lm_head
    config = full_model.config.text_config
    config.model_type = "qwen3_5"

    llm = LLMWrapper(text_model, lm_head, config)
    logger.info(f"Extracted LLM: text_model={text_model.__class__.__name__}, layers={len(text_model.layers)}")

    from auto_gptq import BaseQuantizeConfig
    from auto_gptq.modeling.qwen3_5 import Qwen3_5GPTQForCausalLM

    quantize_config = BaseQuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
    )
    logger.info(f"Quantize config: bits={BITS}, group_size={GROUP_SIZE}")

    gptq_model = Qwen3_5GPTQForCausalLM(llm, False, quantize_config)

    logger.info("Starting GPTQ quantization...")
    gptq_model.quantize(examples, batch_size=1)
    logger.info("Quantization complete!")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    llm_temp_dir = os.path.join(OUTPUT_PATH, "_llm_temp")
    os.makedirs(llm_temp_dir, exist_ok=True)
    gptq_model.save_quantized(llm_temp_dir, use_safetensors=True)
    logger.info(f"Saved quantized LLM to temp dir")

    for fname in os.listdir(MODEL_PATH):
        src = os.path.join(MODEL_PATH, fname)
        dst = os.path.join(OUTPUT_PATH, fname)
        if fname.startswith("model") and fname.endswith(".safetensors"):
            continue
        if fname in ("config.json", "quantize_config.json", "model.safetensors.index.json"):
            continue
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            logger.info(f"  Copied: {fname}")
        elif os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
            logger.info(f"  Copied dir: {fname}")

    logger.info("Building combined state dict...")
    quantized_llm_files = [f for f in os.listdir(llm_temp_dir) if f.endswith(".safetensors")]
    llm_state_dict = {}
    for f in quantized_llm_files:
        sd = safe_load(os.path.join(llm_temp_dir, f))
        llm_state_dict.update(sd)
    logger.info(f"  Loaded {len(llm_state_dict)} quantized LLM tensors")

    non_llm_state_dict = {}
    full_model_sd = full_model.state_dict()
    for key, value in full_model_sd.items():
        if not key.startswith("model.language_model.") and not key.startswith("lm_head."):
            non_llm_state_dict[key] = value
    logger.info(f"  Collected {len(non_llm_state_dict)} non-LLM tensors (vision, merger, etc.)")

    combined_sd = {}
    for key, value in llm_state_dict.items():
        if key.startswith("lm_head."):
            logger.info(f"  Skipping tied weight: {key}")
            continue
        if key.startswith("model."):
            combined_sd[f"model.language_model.{key[len('model.'):]}"] = value
        else:
            combined_sd[f"model.language_model.{key}"] = value
    for key, value in non_llm_state_dict.items():
        combined_sd[key] = value
    logger.info(f"  Total combined tensors: {len(combined_sd)}")

    MAX_SHARD_SIZE_BYTES = 4 * 1024 * 1024 * 1024
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
        logger.info(f"  Saved single shard: {shard_name}")
    else:
        for i, shard in enumerate(shards):
            shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            safe_save(shard, os.path.join(OUTPUT_PATH, shard_name))
            for key in shard:
                weight_map[key] = shard_name
            logger.info(f"  Saved shard {i+1}/{len(shards)}: {shard_name}")

    index = {
        "metadata": {"total_size": sum(t.nelement() * t.element_size() for t in combined_sd.values())},
        "weight_map": weight_map,
    }
    with open(os.path.join(OUTPUT_PATH, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    config_full = AutoConfig.from_pretrained(MODEL_PATH)
    config_dict = config_full.to_dict()
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
        "modules_in_block_to_quantize": [
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.o_proj"],
            ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
            ["linear_attn.out_proj"],
            ["mlp.gate_proj", "mlp.up_proj"],
            ["mlp.down_proj"],
        ],
    }

    with open(os.path.join(OUTPUT_PATH, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("  Saved config.json with quantization_config")

    quant_config_dict = quantize_config.to_dict()
    with open(os.path.join(OUTPUT_PATH, "quantize_config.json"), "w") as f:
        json.dump(quant_config_dict, f, indent=2)

    shutil.rmtree(llm_temp_dir)
    logger.info("  Cleaned up temp directory")

    del combined_sd, llm_state_dict, non_llm_state_dict, full_model_sd
    torch.cuda.empty_cache()

    logger.info(f"Done! Quantized model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
