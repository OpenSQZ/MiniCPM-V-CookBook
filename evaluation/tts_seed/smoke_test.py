#!/usr/bin/env python3
"""
TTS C++ 评测 Smoke Test —— 端到端跑几条样本验证推理入口是否正常。

做的事：
  1. 从 meta.lst 取前 N 条样本
  2. 调用 generate_cpp.py（prompt_bundle 提取 → manifest 生成 → 调 llama-omni-tts-eval 批量生成 wav）
  3. 检查输出 wav 是否生成、时长是否合理

用法（先 `source pipeline.env`，或用环境变量覆盖路径/设备）：
    source pipeline.env
    python smoke_test.py 3
无显卡时 pipeline.env 默认 CPU（CPP_NGL=0 / T2W_DEVICE=cpu / TTS_DEVICE=cpu），慢但能跑。

所有路径/设备均从环境变量读取（见 pipeline.env）；未 source 时用文档里的示例默认值。
"""

import os
import subprocess
import sys
import wave
import contextlib

WORKDIR = os.path.dirname(os.path.abspath(__file__))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3

# 全部从环境变量读取（pipeline.env 会设好）；未设置时给文档里的示例默认
MODEL_PATH     = os.environ.get("MODEL_PATH", "/path/to/MiniCPM-o-4_5-gguf")
TTS_MODEL_PATH = os.environ.get("TTS_MODEL_PATH", os.path.join(MODEL_PATH, "tts/MiniCPM-o-4_5-tts-F16.gguf"))
CPP_BIN        = os.environ.get("CPP_BIN", "/path/to/llama.cpp-omni/build/bin/llama-omni-tts-eval")
ONNX_MODEL_DIR = os.environ.get("ONNX_MODEL_DIR", "/path/to/Step-Audio-2-mini/token2wav")
EVAL_META_PATH = os.environ.get("EVAL_META_PATH", "/path/to/seedtts_testset_zh/zh/meta.lst")
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH", "/path/to/seedtts_testset_zh/zh")
SEED           = os.environ.get("SEED", "42")

# 设备
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cpu")   # prompt_bundle 提取
T2W_DEVICE = os.environ.get("T2W_DEVICE", "cpu")   # C++ Token2Wav
CPP_NGL    = os.environ.get("CPP_NGL", "0")        # C++ LLM offload 层数

# 运行时库：C++ 构建目录（含 libomni.so 等）+ 可选 CUDA 运行库
CPP_BUILD_LIB    = os.environ.get("CPP_BUILD_LIB", os.path.dirname(CPP_BIN))
CUDA_RUNTIME_LIB = os.environ.get("CUDA_RUNTIME_LIB", "")

SAVE_DIR = os.path.join(WORKDIR, "eval_results", f"smoke-{N}")


def wav_duration(path):
    with contextlib.closing(wave.open(path, "rb")) as w:
        return w.getnframes() / float(w.getframerate())


def main():
    print(f"=== TTS Smoke Test: {N} samples ===")
    print(f"  MODEL_PATH : {MODEL_PATH}")
    print(f"  CPP_BIN    : {CPP_BIN}")
    print(f"  device     : ngl={CPP_NGL} t2w={T2W_DEVICE} extract={TTS_DEVICE}")
    print(f"  SAVE_DIR   : {SAVE_DIR}")

    if not os.path.exists(CPP_BIN):
        print(f"[FAIL] C++ binary not found: {CPP_BIN}  (见 README 编译 llama-omni-tts-eval)")
        sys.exit(1)

    os.makedirs(SAVE_DIR, exist_ok=True)

    env = os.environ.copy()
    ld = [p for p in (CUDA_RUNTIME_LIB, CPP_BUILD_LIB, env.get("LD_LIBRARY_PATH", "")) if p]
    env["LD_LIBRARY_PATH"] = ":".join(ld)

    cmd = [
        sys.executable, os.path.join(WORKDIR, "generate_cpp.py"),
        "--cpp-bin", CPP_BIN,
        "--model-path", MODEL_PATH,
        "--tts-model-path", TTS_MODEL_PATH,
        "--save-dir", SAVE_DIR,
        "--eval-meta-path", EVAL_META_PATH,
        "--eval-data-path", EVAL_DATA_PATH,
        "--onnx-model-dir", ONNX_MODEL_DIR,
        "--language", "zh",
        "--seed", SEED,
        "--temperature", "0.3",
        "--teacher-forcing",
        "--device", TTS_DEVICE,
        "--n-gpu-layers", CPP_NGL,
        "--t2w-device", T2W_DEVICE,
        "--num-samples", str(N),
        "--rank", "0", "--world-size", "1",
    ]
    print("  Running:", " ".join(cmd))
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        print(f"[FAIL] generate_cpp.py exited with {ret.returncode}")
        sys.exit(1)

    # --- 验证输出 ---
    print("\n=== Verify generated wavs ===")
    samples = []
    with open(EVAL_META_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            utt = parts[0]
            if utt.endswith(".wav"):
                utt = utt[:-4]
            samples.append(utt)
            if len(samples) >= N:
                break

    ok = 0
    for utt in samples:
        wav = os.path.join(SAVE_DIR, f"{utt}.wav")
        if os.path.exists(wav):
            try:
                dur = wav_duration(wav)
            except Exception as e:
                print(f"  [BAD ] {utt}.wav unreadable: {e}")
                continue
            flag = "OK " if 0.3 < dur < 79.0 else "WARN"
            print(f"  [{flag}] {utt}.wav  {dur:.2f}s")
            if flag == "OK ":
                ok += 1
        else:
            print(f"  [MISS] {utt}.wav not generated")

    print(f"\n=== Smoke test: {ok}/{len(samples)} wav 生成且时长合理 ===")
    sys.exit(0 if ok == len(samples) else 1)


if __name__ == "__main__":
    main()
