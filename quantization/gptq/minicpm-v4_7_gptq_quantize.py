"""GPTQ 4-bit quantization for MiniCPM-V 4.7 (AutoGPTQ fork tc-mb/AutoGPTQ).

Extracts the Qwen3.5 text backbone from a MiniCPM-V 4.7 checkpoint, quantizes it
with GPTQ, and reassembles the full multimodal model with the original vision
components. Only the dense 1B backbone is supported (35B-A3B MoE uses fused
expert parameters that AutoGPTQ cannot quantize).

Usage:
  python gptq_quantize.py --model-path <hf_dir> --save-path <out_dir>
"""

import argparse
import json
import logging
import os
import shutil

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

BITS = 4
GROUP_SIZE = 128
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LEN = 256
MAX_SHARD_SIZE_BYTES = 4 * 1024 * 1024 * 1024


def prepare_calibration_data(tokenizer, num_samples, max_length):
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    examples = []
    for sample in dataset:
        text = sample.get("text", "") or sample.get("output", "") or sample.get("instruction", "")
        if not text:
            continue
        tokenized = tokenizer(
            text, truncation=True, max_length=max_length, return_tensors="pt",
        )
        examples.append({
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        })
    logger.info(f"Prepared {len(examples)} calibration samples")
    return examples


class LLMWrapper(torch.nn.Module):
    """Wraps the Qwen3.5 text model + lm_head to look like a CausalLM for GPTQ."""

    def __init__(self, text_model, lm_head, config):
        super().__init__()
        self.model = text_model
        self.lm_head = lm_head
        self.config = config
        self.seqlen = MAX_SEQ_LEN

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs[0]
        return self.lm_head(hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--save-path", required=True)
    args = ap.parse_args()

    MODEL_PATH = args.model_path
    OUTPUT_PATH = args.save_path

    logger.info(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    logger.info("Preparing calibration data...")
    examples = prepare_calibration_data(tokenizer, NUM_CALIBRATION_SAMPLES, MAX_SEQ_LEN)

    logger.info(f"Loading full model from {MODEL_PATH}")
    full_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="cpu", trust_remote_code=True,
    )

    text_model = full_model.model.language_model
    lm_head = full_model.lm_head
    config = full_model.config.text_config
    config.model_type = "qwen3_5"

    llm = LLMWrapper(text_model, lm_head, config)
    logger.info(f"Extracted LLM: {text_model.__class__.__name__}, layers={len(text_model.layers)}")

    from auto_gptq import BaseQuantizeConfig
    from auto_gptq.modeling.qwen3_5 import Qwen3_5GPTQForCausalLM

    quantize_config = BaseQuantizeConfig(
        bits=BITS, group_size=GROUP_SIZE, desc_act=False, sym=True,
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

    logger.info("Building combined state dict...")
    llm_state_dict = {}
    for f in os.listdir(llm_temp_dir):
        if f.endswith(".safetensors"):
            llm_state_dict.update(safe_load(os.path.join(llm_temp_dir, f)))
    logger.info(f"  Loaded {len(llm_state_dict)} quantized LLM tensors")

    non_llm_state_dict = {}
    full_model_sd = full_model.state_dict()
    for key, value in full_model_sd.items():
        if not key.startswith("model.language_model.") and not key.startswith("lm_head."):
            non_llm_state_dict[key] = value
    logger.info(f"  Collected {len(non_llm_state_dict)} non-LLM tensors")

    combined_sd = {}
    for key, value in llm_state_dict.items():
        if key.startswith("lm_head."):
            continue
        if key.startswith("model."):
            combined_sd[f"model.language_model.{key[len('model.'):]}"] = value
        else:
            combined_sd[f"model.language_model.{key}"] = value
    for key, value in non_llm_state_dict.items():
        combined_sd[key] = value
    logger.info(f"  Total combined tensors: {len(combined_sd)}")

    sorted_keys = sorted(combined_sd.keys())
    shards, current_shard, current_size, weight_map = [], {}, 0, {}
    for key in sorted_keys:
        tensor = combined_sd[key]
        tsize = tensor.nelement() * tensor.element_size()
        if current_size > 0 and current_size + tsize > MAX_SHARD_SIZE_BYTES:
            shards.append(current_shard)
            current_shard, current_size = {}, 0
        current_shard[key] = tensor.clone().contiguous()
        current_size += tsize
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

    config_full = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
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

    with open(os.path.join(OUTPUT_PATH, "quantize_config.json"), "w") as f:
        json.dump(quantize_config.to_dict(), f, indent=2)

    shutil.rmtree(llm_temp_dir, ignore_errors=True)
    logger.info(f"Done! Quantized model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
