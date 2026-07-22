# Daily-Omni C++ 评测

该流水线为每张 GPU 启动一个常驻 `llama-omni-eval-daily-cli` 进程，完成视频帧采样、音频交错输入、失败题重跑和结果评分。

## 准备

```bash
pip install pandas python-dotenv decord Pillow soundfile
cd /path/to/llama.cpp-omni
git apply /path/to/MiniCPM-V-CookBook/evaluation/daily-omni/llama-omni-eval-daily-cli.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-omni-eval-daily-cli -j
```

将 `.env.example` 复制为 `.env`，填写 CLI、GGUF 模型和 Daily-Omni 数据集路径。GGUF 模型从 [openbmb/MiniCPM-o-4_5-gguf](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf) 下载；数据集需包含 JSONL 标注及其中引用的媒体文件。

## 运行

```bash
python smoke_test.py 2
python eval_cpp_pipeline.py --num-gpus 1 --limit 6
python eval_cpp_pipeline.py --num-gpus 8
```

可用 `--skip-rerun` 或 `--skip-scoring` 跳过重跑或评分。生成的结果在本地 `output/` 下，Git 不会跟踪。

