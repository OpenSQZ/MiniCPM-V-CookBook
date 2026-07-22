#!/usr/bin/env python3
"""
CPP 版 MiniCPM-o 4.5 TTS 测评 — Python 调度脚本

职责：
  1. 解析 meta.lst（兼容 2/3/4/5 字段格式，与 generate_o45.py 对齐）
  2. 去重所有 prompt_wav_path，批量调用 extract_prompt_bundle.py 提取 prompt_bundle
  3. 生成 C++ 输入清单 manifest.tsv
  4. 调用 C++ 推理程序 llama-omni-tts-eval
  5. 支持多 GPU 并行（--rank / --world_size，数据交错切分）
  6. 支持断点续传（跳过已生成的 wav）

用法:
    python generate_cpp.py \
        --cpp-bin /path/to/llama-omni-tts-eval \
        --model-path /path/to/model_dir \
        --save-dir /path/to/output \
        --eval-meta-path /path/to/meta.lst \
        --eval-data-path /path/to/zh \
        --onnx-model-dir /path/to/onnx_models \
        --language zh \
        --teacher-forcing \
        --rank 0 --world-size 1
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time

from tqdm import tqdm


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text, char="=", width=80):
    text = f" {text} "
    padding = (width - len(text)) // 2
    line = char * padding + text + char * padding
    if len(line) < width:
        line += char
    print(f"\n{Colors.HEADER}{Colors.BOLD}{line}{Colors.ENDC}")


def print_info(text):
    print(f"  {Colors.OKCYAN}[INFO] {text}{Colors.ENDC}")


def print_success(text):
    print(f"  {Colors.OKGREEN}[OK]   {text}{Colors.ENDC}")


def print_warning(text):
    print(f"  {Colors.WARNING}[WARN] {text}{Colors.ENDC}")


def print_error(text):
    print(f"  {Colors.FAIL}[ERR]  {text}{Colors.ENDC}")


# ==================== meta.lst 解析 ====================

def parse_meta_lst(meta_path):
    """
    解析 meta.lst，兼容 2/3/4/5 字段格式。
    返回列表: [(utt, prompt_text, prompt_wav_path, infer_text), ...]
    prompt_wav_path 可能为 None（2 字段格式）。
    """
    samples = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            utt = None
            prompt_text = None
            prompt_wav_path = None
            infer_text = None

            if len(parts) == 5:
                utt, prompt_text, prompt_wav_path, infer_text, _infer_wav = parts
            elif len(parts) == 4:
                utt, prompt_text, prompt_wav_path, infer_text = parts
            elif len(parts) == 2:
                utt, infer_text = parts
            elif len(parts) == 3:
                utt, infer_text, prompt_wav_path = parts
                if utt.endswith(".wav"):
                    utt = utt[:-4]
            else:
                print_warning(f"Skipping malformed line: {line}")
                continue

            samples.append((utt, prompt_text, prompt_wav_path, infer_text))
    return samples


def wav_path_hash(wav_path):
    """对 wav 路径取 MD5 前 12 位作为 bundle 子目录名。"""
    return hashlib.md5(wav_path.encode()).hexdigest()[:12]


# ==================== prompt_bundle 批量提取 ====================

def prepare_prompt_bundles(samples, eval_data_path, bundle_base_dir,
                           onnx_model_dir, device="cuda", skip_existing=True):
    """
    扫描所有样本的 prompt_wav_path，去重后批量提取 prompt_bundle。
    返回 dict: {relative_wav_path: bundle_dir}
    """
    unique_wavs = set()
    for utt, prompt_text, prompt_wav_path, infer_text in samples:
        if prompt_wav_path:
            unique_wavs.add(prompt_wav_path)

    print_info(f"Total unique prompt wavs: {len(unique_wavs)}")

    os.makedirs(bundle_base_dir, exist_ok=True)

    wav_to_bundle = {}
    batch_list_path = os.path.join(bundle_base_dir, "_batch_list.tsv")

    tasks_to_extract = []
    for wav_rel in sorted(unique_wavs):
        wav_full = os.path.join(eval_data_path, wav_rel)
        h = wav_path_hash(wav_rel)
        out_dir = os.path.join(bundle_base_dir, h)
        wav_to_bundle[wav_rel] = out_dir

        if skip_existing and os.path.exists(os.path.join(out_dir, "spk_f32.bin")):
            continue
        tasks_to_extract.append((wav_full, out_dir))

    if not tasks_to_extract:
        print_success("All prompt_bundles already extracted, skipping.")
        return wav_to_bundle

    print_info(f"Need to extract {len(tasks_to_extract)} prompt_bundles "
               f"({len(unique_wavs) - len(tasks_to_extract)} already cached)")

    with open(batch_list_path, "w") as f:
        for wav_full, out_dir in tasks_to_extract:
            f.write(f"{wav_full}\t{out_dir}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_script = os.path.join(script_dir, "extract_prompt_bundle.py")

    cmd = [
        sys.executable, extract_script,
        "--batch-list", batch_list_path,
        "--model-dir", onnx_model_dir,
        "--device", device,
        "--skip-existing",
    ]
    print_info(f"Running: {' '.join(cmd)}")
    t0 = time.time()
    ret = subprocess.run(cmd)
    elapsed = time.time() - t0

    if ret.returncode != 0:
        print_error(f"extract_prompt_bundle.py failed with code {ret.returncode}")
        sys.exit(1)

    print_success(f"Prompt bundle extraction done in {elapsed:.1f}s")
    return wav_to_bundle


# ==================== manifest 生成 ====================

def generate_manifest(samples, eval_data_path, wav_to_bundle, save_dir,
                      manifest_path, skip_existing=True):
    """
    生成 C++ 输入 manifest.tsv。
    每行: ref_audio_path \t bundle_dir \t infer_text \t output_wav_path
    """
    skipped = 0
    written = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for utt, prompt_text, prompt_wav_path, infer_text in samples:
            output_wav = os.path.join(save_dir, f"{utt}.wav")

            if skip_existing and os.path.exists(output_wav):
                skipped += 1
                continue

            if prompt_wav_path is None:
                print_warning(f"No prompt_wav for {utt}, skipping")
                continue

            ref_audio_full = os.path.join(eval_data_path, prompt_wav_path)
            bundle_dir = wav_to_bundle.get(prompt_wav_path, "")

            if not bundle_dir or not os.path.exists(
                    os.path.join(bundle_dir, "spk_f32.bin")):
                print_warning(f"Missing bundle for {prompt_wav_path}, skipping {utt}")
                continue

            f.write(f"{ref_audio_full}\t{bundle_dir}\t{infer_text}\t{output_wav}\n")
            written += 1

    print_info(f"Manifest: {written} samples to process, {skipped} skipped (already exist)")
    return written


# ==================== C++ 推理调用 ====================

def run_cpp_inference(cpp_bin, model_path, manifest_path, language="zh",
                      teacher_forcing=True, seed=42, temperature=0.3,
                      tts_model_path=None, n_gpu_layers=99, t2w_device="gpu:0"):
    """调用 C++ 推理程序处理 manifest 中的所有样本。"""
    if not os.path.exists(cpp_bin):
        print_error(f"C++ binary not found: {cpp_bin}")
        print_info("请先编译 llama-omni-tts-eval，参见 README/CLI-Migration.md")
        sys.exit(1)

    cmd = [
        cpp_bin,
        "-m", model_path,
        "--manifest", manifest_path,
        "--language", language,
        "--seed", str(seed),
        "--temperature", str(temperature),
        "-ngl", str(n_gpu_layers),
        "--t2w-device", t2w_device,
    ]
    if teacher_forcing:
        cmd.append("--teacher-forcing")
    if tts_model_path:
        cmd.extend(["--tts", tts_model_path])

    print_info(f"Running C++ inference: {' '.join(cmd)}")
    t0 = time.time()
    ret = subprocess.run(cmd)
    elapsed = time.time() - t0

    if ret.returncode != 0:
        print_error(f"C++ inference failed with code {ret.returncode}")
        return False

    print_success(f"C++ inference done in {elapsed:.1f}s")
    return True


# ==================== 主函数 ====================

def main(args):
    print_header("CPP TTS Evaluation - Generate Stage")

    print_info(f"meta_path:      {args.eval_meta_path}")
    print_info(f"eval_data_path: {args.eval_data_path}")
    print_info(f"save_dir:       {args.save_dir}")
    print_info(f"model_path:     {args.model_path}")
    print_info(f"tts_model_path: {args.tts_model_path or '(default from model_dir)'}")
    print_info(f"rank:           {args.rank}/{args.world_size}")
    print_info(f"language:       {args.language}")
    print_info(f"teacher_forcing:{args.teacher_forcing}")
    print_info(f"seed:           {args.seed}")

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Step 1: 解析 meta.lst ---
    print_header("Step 1: Parse meta.lst", char="-", width=60)
    all_samples = parse_meta_lst(args.eval_meta_path)
    print_info(f"Total samples in meta.lst: {len(all_samples)}")

    # 按 rank 切分（交错方式，与 generate_o45.py 一致）
    my_samples = all_samples[args.rank::args.world_size]
    print_info(f"Samples for rank {args.rank}: {len(my_samples)}")

    if args.num_samples < len(my_samples):
        my_samples = my_samples[:args.num_samples]
        print_info(f"Truncated to {args.num_samples} samples")

    # --- Step 2: 提取 prompt_bundle（全量去重，只在 rank 0 时执行或所有 rank 独立执行） ---
    print_header("Step 2: Extract prompt_bundles", char="-", width=60)
    bundle_base_dir = os.path.join(args.save_dir, "_prompt_bundles")
    wav_to_bundle = prepare_prompt_bundles(
        my_samples, args.eval_data_path, bundle_base_dir,
        args.onnx_model_dir, device=args.device,
        skip_existing=True,
    )

    # --- Step 3: 生成 manifest ---
    print_header("Step 3: Generate manifest", char="-", width=60)
    manifest_path = os.path.join(args.save_dir, f"manifest_rank{args.rank}.tsv")
    n_todo = generate_manifest(
        my_samples, args.eval_data_path, wav_to_bundle,
        args.save_dir, manifest_path, skip_existing=(not args.no_skip),
    )

    if n_todo == 0:
        print_success("All samples already generated, nothing to do.")
        return

    # --- Step 4: 调用 C++ 推理 ---
    print_header("Step 4: C++ Inference", char="-", width=60)
    ok = run_cpp_inference(
        args.cpp_bin, args.model_path, manifest_path,
        language=args.language,
        teacher_forcing=args.teacher_forcing,
        seed=args.seed,
        temperature=args.temperature,
        tts_model_path=args.tts_model_path,
        n_gpu_layers=args.n_gpu_layers,
        t2w_device=args.t2w_device,
    )

    # --- Step 5: 验证输出 ---
    print_header("Step 5: Verify outputs", char="-", width=60)
    total = len(my_samples)
    generated = sum(
        1 for utt, _, _, _ in my_samples
        if os.path.exists(os.path.join(args.save_dir, f"{utt}.wav"))
    )
    missing = total - generated
    print_info(f"Generated: {generated}/{total}")
    if missing > 0:
        print_warning(f"Missing: {missing} samples")
        fail_log = os.path.join(args.save_dir, f"failed_rank{args.rank}.txt")
        with open(fail_log, "w") as f:
            for utt, _, _, infer_text in my_samples:
                wav = os.path.join(args.save_dir, f"{utt}.wav")
                if not os.path.exists(wav):
                    f.write(f"{utt}\t{infer_text}\n")
        print_info(f"Failed list saved to {fail_log}")
    else:
        print_success("All samples generated successfully!")

    print_header("Generate Stage Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CPP TTS eval: data processing + prompt_bundle extraction + C++ invocation"
    )

    # C++ 相关（默认从环境变量读取，见 pipeline.env；否则用文档里的示例路径）
    parser.add_argument("--cpp-bin", type=str,
                        default=os.environ.get(
                            "CPP_BIN",
                            "/path/to/llama.cpp-omni/build/bin/llama-omni-tts-eval"),
                        help="Path to compiled llama-omni-tts-eval binary (see README to build)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to GGUF model directory (for C++ inference)")

    # 数据
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory to save output wav files")
    parser.add_argument("--eval-meta-path", type=str, required=True,
                        help="Path to meta.lst")
    parser.add_argument("--eval-data-path", type=str, required=True,
                        help="Path to evaluation data dir (parent of prompt-wavs/)")

    # prompt_bundle 提取
    parser.add_argument("--onnx-model-dir", type=str,
                        default=os.environ.get(
                            "ONNX_MODEL_DIR",
                            "/path/to/Step-Audio-2-mini/token2wav"),
                        help="Dir containing speech_tokenizer_v2_25hz.onnx and campplus.onnx")
    parser.add_argument("--device", type=str,
                        default=os.environ.get("TTS_DEVICE", "cuda"),
                        help="Device for prompt_bundle extraction (cuda/cpu)")

    # TTS 模型路径覆盖（用于测试量化版本等）
    parser.add_argument("--tts-model-path", type=str, default=None,
                        help="Override TTS GGUF model path (passed to C++ --tts)")

    # 推理参数
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num-samples", type=int, default=10000000)
    parser.add_argument("--teacher-forcing", action="store_true")
    # 设备：C++ LLM offload 层数与 Token2Wav 设备（CPU 跑用 -ngl 0 / cpu）
    parser.add_argument("--n-gpu-layers", type=int,
                        default=int(os.environ.get("CPP_NGL", "99")),
                        help="C++ LLM GPU offload layers (0 = all CPU)")
    parser.add_argument("--t2w-device", type=str,
                        default=os.environ.get("T2W_DEVICE", "gpu:0"),
                        help="C++ Token2Wav device (gpu:0 / cpu)")

    # 并行
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)

    # 续传
    parser.add_argument("--no-skip", action="store_true",
                        help="Do not skip already generated wavs (re-generate all)")

    args = parser.parse_args()
    main(args)
