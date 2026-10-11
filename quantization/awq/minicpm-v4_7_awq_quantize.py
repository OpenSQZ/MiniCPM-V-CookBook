"""AWQ 4-bit quantization for MiniCPM-V 4.7 (AutoAWQ fork tc-mb/AutoAWQ).

Usage:
  python awq_quantize.py --model-path <hf_dir> --save-path <out_dir>
"""

import argparse
import os
import shutil

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM


def copy_files_not_in_B(A_path, B_path):
    if not os.path.exists(A_path):
        raise FileNotFoundError(f"The directory {A_path} does not exist.")
    os.makedirs(B_path, exist_ok=True)

    files_in_A = {
        f for f in os.listdir(A_path)
        if not (f.endswith(".bin") or f.endswith(".safetensors"))
    }
    files_in_B = set(os.listdir(B_path))

    for f in files_in_A - files_in_B:
        src = os.path.join(A_path, f)
        dst = os.path.join(B_path, f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)


def build_calib(tokenizer, n=1024):
    data = load_dataset("tatsu-lab/alpaca", split="train")

    def concatenate(x):
        if x["input"] and x["instruction"]:
            msgs = [
                {"role": "system", "content": x["instruction"]},
                {"role": "user", "content": x["input"]},
                {"role": "assistant", "content": x["output"]},
            ]
        elif x["input"]:
            msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": x["input"]},
                {"role": "assistant", "content": x["output"]},
            ]
        else:
            msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": x["instruction"]},
                {"role": "assistant", "content": x["output"]},
            ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {"text": text}

    return [r["text"] for r in data.map(concatenate)][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--w-bit", type=int, default=4)
    ap.add_argument("--q-group-size", type=int, default=128)
    ap.add_argument("--calib-samples", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM",
    }

    print(f"[awq] loading {args.model_path}", flush=True)
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model.model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("[awq] building calibration data", flush=True)
    calib_data = build_calib(tokenizer, args.calib_samples)

    print("[awq] quantizing", flush=True)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    print(f"[awq] saving to {args.save_path}", flush=True)
    model.save_quantized(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    copy_files_not_in_B(args.model_path, args.save_path)
    print("[awq] done", flush=True)


if __name__ == "__main__":
    main()
