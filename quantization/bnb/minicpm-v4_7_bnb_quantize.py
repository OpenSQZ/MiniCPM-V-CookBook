"""bitsandbytes 4-bit quantization for MiniCPM-V 4.7.

Loads an HF-format MiniCPM-V 4.7 checkpoint (weights + trust_remote_code code),
quantizes to NF4, runs a sanity inference, and saves the quantized model.
"""

import argparse
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="What is in this picture?")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda:0", help="'cuda:N' or 'auto' for multi-GPU")
    ap.add_argument("--no-sanity", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["lm_head"],
        llm_int8_threshold=6.0,
    )

    print(f"[bnb] loading processor from {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"[bnb] loading model from {args.model_path}", flush=True)
    t0 = time.time()
    device_map = "auto" if args.device == "auto" else {"": args.device}
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        device_map=device_map,
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    print(f"[bnb] loaded in {time.time() - t0:.1f}s", flush=True)

    if not args.no_sanity and args.image:
        image = Image.open(args.image).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        t0 = time.time()
        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        response = processor.decode(
            out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        print(f"[bnb] sanity output ({time.time() - t0:.1f}s): {response}", flush=True)

    os.makedirs(args.save_path, exist_ok=True)
    print(f"[bnb] saving to {args.save_path}", flush=True)
    model.save_pretrained(args.save_path, safe_serialization=True)
    processor.save_pretrained(args.save_path)
    print("[bnb] done", flush=True)


if __name__ == "__main__":
    main()
