# Daily-Omni C++ Evaluation

This pipeline evaluates MiniCPM-o on Daily-Omni through a persistent `llama-omni-eval-daily-cli` process per GPU. It samples video frames, interleaves audio segments, supports failed-item retries, and scores the resulting JSON.

## Setup

Install Python dependencies:

```bash
pip install pandas python-dotenv decord Pillow soundfile
```

Apply [llama-omni-eval-daily-cli.patch](llama-omni-eval-daily-cli.patch) to a compatible `llama.cpp-omni` checkout, then build its target:

```bash
cd /path/to/llama.cpp-omni
git apply /path/to/MiniCPM-V-CookBook/evaluation/daily-omni/llama-omni-eval-daily-cli.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-omni-eval-daily-cli -j
```

Create `.env` from `.env.example` and provide the CLI executable, GGUF model, and Daily-Omni dataset paths. Download the MiniCPM-o GGUF model from [openbmb/MiniCPM-o-4_5-gguf](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf). The dataset must provide the JSONL annotation file and media paths referenced by it.

## Run

```bash
python smoke_test.py 2
python eval_cpp_pipeline.py --num-gpus 1 --limit 6
python eval_cpp_pipeline.py --num-gpus 8
```

Use `--skip-rerun` or `--skip-scoring` for inference-only runs. Outputs are written locally under `output/` and are intentionally ignored by Git.
