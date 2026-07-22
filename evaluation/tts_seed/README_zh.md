# MiniCPM-o C++ TTS 评测

本目录用于生成 Seed-TTS 中文测试集语音，并计算 WER 与说话人相似度。仓库仅保留流水线和指标代码；数据集、模型权重、生成音频及结果均不入库。

## 准备

```bash
cd /path/to/llama.cpp-omni
git apply /path/to/MiniCPM-V-CookBook/evaluation/tts_seed/llama-omni-tts-eval.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-omni-tts-eval -j

pip install torch torchaudio tqdm numpy scipy soundfile librosa jiwer zhon zhconv \
  onnxruntime s3tokenizer funasr transformers sentencepiece
git clone https://github.com/s3prl/s3prl.git /path/to/s3prl
```

在 `pipeline.env` 中填写所有外部路径，也可在运行前设置同名环境变量。需要自行下载：

- MiniCPM-o GGUF：[openbmb/MiniCPM-o-4_5-gguf](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf)。
- prompt bundle ONNX： [stepfun-ai/Step-Audio-2-mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini) 的 `token2wav/speech_tokenizer_v2_25hz.onnx` 与 `campplus.onnx`。
- 中文 WER 模型：[funasr/paraformer-zh](https://huggingface.co/funasr/paraformer-zh)。
- 可选 SIM 权重：[wavlm_large_finetune.pth](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view)。
- Seed-TTS 中文测试集（含 `meta.lst` 与其中引用的 prompt 音频）。

将 `S3PRL_REPO` 指向单独 clone 的 s3prl。仓库不包含任何模型权重或软链接。

## 运行

```bash
source pipeline.env
python smoke_test.py 3
bash run_tts_eval_cpp_zh.sh
bash run_eval_only.sh /path/to/generated-wavs
```

`SPEAKER_CKPT` 可不设置；此时会完成生成和 WER，跳过 SIM。生成音频、日志和结果仅保存在本地忽略目录中。

