# Video-MME C++ Evaluation

This pipeline evaluates MiniCPM-o on Video-MME through a persistent `llama-omni-eval-cli` process per GPU. It samples frames, retries malformed responses, and runs the bundled Video-MME scorer.

## Setup

```bash
pip install pandas pyarrow python-dotenv decord Pillow
cd /path/to/llama.cpp-omni
git apply /path/to/MiniCPM-V-CookBook/evaluation/videomme/llama-omni-eval-cli.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-omni-eval-cli -j
```

Copy `.env.example` to `.env` and set the CLI executable, GGUF model, Video-MME parquet file, and video directory. Download the MiniCPM-o GGUF files from [openbmb/MiniCPM-o-4_5-gguf](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf). Obtain Video-MME data separately and point `PARQUET_PATH` and `VIDEO_DATA_DIR` to it.

## Run

```bash
python smoke_test.py 2
python eval_cpp_pipeline.py --num-gpus 1 --limit 3 --skip-rerun --skip-scoring
python eval_cpp_pipeline.py --num-gpus 8
```

Results are local runtime artifacts under `output/` and are ignored by Git.
