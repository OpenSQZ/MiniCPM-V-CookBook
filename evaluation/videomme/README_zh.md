# Video-MME C++ 评测

该流水线为每张 GPU 启动一个常驻 `llama-omni-eval-cli` 进程，完成视频帧采样、异常回答重跑和 Video-MME 评分。

## 准备

```bash
pip install pandas pyarrow python-dotenv decord Pillow
cd /path/to/llama.cpp-omni
git apply /path/to/MiniCPM-V-CookBook/evaluation/videomme/llama-omni-eval-cli.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-omni-eval-cli -j
```

将 `.env.example` 复制为 `.env`，填写 CLI、GGUF 模型、Video-MME parquet 和视频目录。GGUF 模型从 [openbmb/MiniCPM-o-4_5-gguf](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf) 下载；Video-MME 数据需单独获取后配置 `PARQUET_PATH` 与 `VIDEO_DATA_DIR`。

## 运行

```bash
python smoke_test.py 2
python eval_cpp_pipeline.py --num-gpus 1 --limit 3 --skip-rerun --skip-scoring
python eval_cpp_pipeline.py --num-gpus 8
```

结果仅作为本地运行产物保存于 `output/`，Git 不会跟踪。

