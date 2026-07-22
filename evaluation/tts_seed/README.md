# MiniCPM-o C++ TTS Evaluation

This directory generates speech for the Chinese Seed-TTS evaluation set and reports WER and speaker similarity. It contains only the pipeline and metric code; datasets, model weights, generated audio, and score files are not versioned.

## Setup

Build `llama-omni-tts-eval` from a compatible `llama.cpp-omni` checkout:

```bash
cd /path/to/llama.cpp-omni
git apply /path/to/MiniCPM-V-CookBook/evaluation/tts_seed/llama-omni-tts-eval.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-omni-tts-eval -j
```

Install the Python dependencies:

```bash
pip install torch torchaudio tqdm numpy scipy soundfile librosa jiwer zhon zhconv \
  onnxruntime s3tokenizer funasr transformers sentencepiece
git clone https://github.com/s3prl/s3prl.git /path/to/s3prl
```

Configure all external locations in `pipeline.env` or export them before launching a script. Required downloads are:

- MiniCPM-o GGUF files: [openbmb/MiniCPM-o-4_5-gguf](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf).
- `speech_tokenizer_v2_25hz.onnx` and `campplus.onnx`: `token2wav/` in [stepfun-ai/Step-Audio-2-mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini).
- Chinese WER model: [funasr/paraformer-zh](https://huggingface.co/funasr/paraformer-zh).
- Optional speaker-similarity checkpoint: [wavlm_large_finetune.pth](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view).
- The Seed-TTS Chinese evaluation set, including `meta.lst` and referenced prompt audio.

Set `S3PRL_REPO` to the separate s3prl checkout. No model weights or symlinks are included in this repository.

## Run

```bash
source pipeline.env
python smoke_test.py 3
bash run_tts_eval_cpp_zh.sh
bash run_eval_only.sh /path/to/generated-wavs
```

`SPEAKER_CKPT` is optional: without it, generation and WER run normally while SIM is skipped. Generated audio, logs, and metrics stay in local ignored directories.
