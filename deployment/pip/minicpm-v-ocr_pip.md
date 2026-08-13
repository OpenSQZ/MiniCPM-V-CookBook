# MiniCPM-V OCR - pip (ocrcpm)

> [!NOTE]
> `ocrcpm` wraps a **pipeline**, not a single call: layout detection runs on CPU, each detected region is cropped, the crops go to an OCR backend one by one, and the results are assembled into Markdown and JSON. Layout detection always runs on CPU; only the OCR model's inference backend is pluggable.
>
> This directory already contains an installable wheel (`ocrcpm-0.1.0-py3-none-any.whl`) and the source used to build it (`wheel_source/`).

> [!IMPORTANT]
> The OCR model is not open-sourced yet; how to obtain it will be documented here once it is released. Every command below uses the placeholder paths `$PWD/layout_model`, `$PWD/ocr_model` and `$PWD/ocr_gguf` for the model directories — substitute your own. Model weights are never packed into the wheel.

## 0. Backend choice and environment

Five inference backends are supported for the OCR model, and their setup cost differs a lot:

| Backend | Setup cost | Notes |
| --- | --- | --- |
| transformers-local | None, works right after the wheel | Simplest, slowest, best for a first run (sections 4 and 5) |
| Ollama | Download the official prebuilt binary | No compilation, uses GGUF, best effort/benefit ratio (section 7) |
| llama.cpp | Build it yourself (~15 min) | Also uses GGUF, one compile step more than Ollama (section 6) |
| SGLang | Pure Python, `pip install -e` | No CUDA extension to compile, high throughput (section 8) |
| vLLM | CUDA extensions must be compiled (~1 h) | High throughput, most work to install (section 9) |

Start with transformers-local as described in sections 2–5 to confirm the model and inputs are fine, then switch backends as needed.

All five backends were compared on the same page (one paper's first page, split into 11 blocks): 10 of the 11 layout blocks came out identical, while the abstract block differed between backends.

Environment requirements:

- Python 3.10–3.12 (verified on 3.12);
- an NVIDIA GPU with at least 24 GB of VRAM. 24 GB is the measured lower bound, and the vLLM backend additionally needs a smaller `--max-num-batched-tokens` as described in section 9;
- a CUDA toolchain (including `bin/nvcc`) for the llama.cpp, SGLang and vLLM backends; Ollama and transformers-local do not need one;
- disk: about 40 GB for models and framework sources, plus ~3 GB of third-party dependencies fetched while building vLLM and ~8 GB for Ollama's model store.

## 1. Prepare a working directory

Create an empty directory and put the wheel, the models and a test file in it:

```bash
mkdir -p ~/ocrcpm_demo
cd ~/ocrcpm_demo
```

The layout should end up as:

```text
ocrcpm_demo/
├── ocrcpm-0.1.0-py3-none-any.whl
├── layout_model/                  # layout detection model (PP-DocLayoutV3, safetensors)
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── ocr_model/                     # OCR model in HuggingFace format
│   ├── config.json
│   ├── modeling_minicpmv.py
│   ├── *.safetensors
│   └── ...
├── ocr_gguf/                      # optional, for the llama.cpp / Ollama backends
│   ├── model-f16.gguf
│   └── mmproj-f16.gguf
└── demo.png
```

Copy both model directories in full — do not copy only the weight files. The config files and the glue code shipped alongside the model are equally required at runtime. The llama.cpp and Ollama backends share the same pair of GGUF files.

None of the inference frameworks live here; they are fetched online: build llama.cpp from upstream source (section 6), download the official Ollama binary (section 7), and get SGLang and vLLM from branches carrying the OCR adaptation (sections 8 and 9).

## 2. Create a Python environment

Python 3.10–3.12 is recommended, in an isolated environment:

```bash
cd ~/ocrcpm_demo
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Some Debian/Ubuntu system Pythons do not bundle the venv module, and `python -m venv` reports `ensurepip is not available`. Install it as prompted (e.g. `apt install python3.12-venv`), or use Conda instead:

```bash
conda create -n ocrcpm-demo python=3.12 -y
conda activate ocrcpm-demo
cd ~/ocrcpm_demo
```

## 3. Install the wheel

```bash
python -m pip install "./ocrcpm-0.1.0-py3-none-any.whl[all]"
```

Check that the console script works:

```bash
ocrcpm --help
```

`[all]` installs the layout parsing, Transformers inference and PDF parsing dependencies, but not vLLM, SGLang or llama.cpp. The core dependency set is deliberately light — only pyyaml, requests and pillow — and everything else is an extra:

| extra | What it installs | Purpose |
| --- | --- | --- |
| `layout` | torch, torchvision, transformers, opencv-headless | Layout parsing |
| `pdf` | pymupdf | PDF input rendering |
| `server` | torch, transformers, accelerate, einops | Bundled HF Transformers OpenAI-compatible server |
| `vllm` | vllm | Local vLLM server backend |
| `sglang` | sglang[all] | Local SGLang server backend |
| `llamacpp` / `ollama` | empty | These backends depend on an external binary, not a Python package |
| `models` | huggingface_hub, modelscope | Model download helpers |
| `all` | layout + pdf + server | No vllm / sglang / llamacpp / ollama |

> The install above needs access to a pip index. For a fully offline setup, supply the dependency wheels up front in addition to the wheel and the models, or hand over a ready-made Conda environment.

### 3.1 Build the wheel from source (optional)

The `wheel_source/` directory here is the complete source behind that wheel: src layout, setuptools build. To repackage after changing the code:

```bash
cd deployment/pip/wheel_source
python -m pip install build
python -m build
```

The artifacts land in `dist/` — a `py3-none-any` wheel plus an sdist. The package is pure Python with no compiled extension, so the wheel is only tens of KB; neither model weights nor inference frameworks are inside it. An editable install also works, so changes take effect immediately:

```bash
python -m pip install -e ".[all]"
```

## 4. Generate the default config

The simplest way to run locally is Transformers, which needs no separate model service. From `~/ocrcpm_demo`:

```bash
ocrcpm init \
  --skip-download \
  --layout-dir "$PWD/layout_model" \
  --ocr-dir "$PWD/ocr_model" \
  --backend transformers-local \
  --layout-device cpu \
  --transformers-device cuda \
  --gpu-id 0 \
  --input "$PWD/demo.png"
```

This downloads nothing. It only checks the local model directories and writes the default config:

```text
~/.config/ocrcpm/config.yaml
```

Once initialized, later `ocrcpm parse` calls do not need `--config`.

The `--input` here is only a placeholder path stored in the config. `ocrcpm init` performs no recognition; what actually gets processed is decided by `ocrcpm parse --input` in section 5, and the two may differ. To use a different GPU, change `--gpu-id 0`. See `ocrcpm init --help` for the full parameter list.

## 5. Run OCR

On an image:

```bash
ocrcpm parse --input "$PWD/demo.png" --output "$PWD/demo.md"
```

On a PDF:

```bash
ocrcpm parse --input "$PWD/demo.pdf" --output "$PWD/demo.md"
```

The command output should contain something like:

```text
"counts": {
  "planned": 1,
  "ok": 1,
  "failed": 0
}
```

The final Markdown is at `~/ocrcpm_demo/demo.md`, and intermediate pipeline results default to `~/ocrcpm_demo/runs/`:

```text
<run_root>/<run_tag>/
  output/*.md               # final Markdown
  results_json/*.json       # bbox / label / text / status per block
  summary.json
  per_page_metrics.{jsonl,csv}
  runtime/
    page_tasks.jsonl
    {layout,page,crop}_timing.jsonl
    intermediate/layouts/*.json
    intermediate/crops/<image_id>/*.png
```

The pipeline is three fixed stages with intermediate results on disk, so any stage can be re-run alone:

```bash
ocrcpm show-config  --config <yaml>   # merged config
ocrcpm parse-layout --config <yaml>   # stage 1: detect blocks, save bbox / label / crops
ocrcpm prepare      --config <yaml>   # stage 2: collect into page_tasks.jsonl
ocrcpm infer        --config <yaml>   # stage 3: per-crop OCR, Markdown assembly
ocrcpm run          --config <yaml>   # full pipeline (1 -> 2 -> 3)
```

`ocrcpm` can also be invoked as `python -m ocrcpm`.

## 6. llama.cpp backend (optional)

> Sections 6–9 each contain an `ocrcpm init`, and all of them write the same `~/.config/ocrcpm/config.yaml`, so the last one wins. Only one backend is active at a time; re-run the `ocrcpm init` of the section you want to go back to. To keep several backend configs side by side, add `--config-output "$PWD/ocrcpm_<backend>.yaml"` to each and select one with `ocrcpm parse --config`.

llama.cpp is not installed through pip; get the source from upstream. The `qwen35` architecture and the `minicpmv4_6` projector this model needs first landed in b10091; this project was verified on b10238, so use that or newer. The converted GGUF files load with stock upstream code and need no patch.

Upstream ships CUDA prebuilts for Windows only — the Linux releases are CPU/Vulkan — so on Linux with NVIDIA you have to build it:

```bash
git clone --depth 1 --branch b10238 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
cd ..
```

To avoid compiling, use the Ollama backend in section 7 instead; it ships ready-made Linux CUDA binaries.

Confirm the server binary exists, then write the config:

```bash
test -x "$PWD/llama.cpp/build/bin/llama-server"

ocrcpm init \
  --skip-download \
  --backend llama-cpp \
  --layout-dir "$PWD/layout_model" \
  --llama-server "$PWD/llama.cpp/build/bin/llama-server" \
  --llama-model "$PWD/ocr_gguf/model-f16.gguf" \
  --llama-mmproj "$PWD/ocr_gguf/mmproj-f16.gguf" \
  --model-name minicpm-v-ocr \
  --layout-device cpu \
  --gpu-id 0 \
  --port 19699 \
  --max-model-len 8192 \
  --input "$PWD/demo.png"
```

Recognition still uses the same command:

```bash
ocrcpm parse --input "$PWD/demo.png" --output "$PWD/demo.md"
```

OCRCPM starts `llama-server`, waits until it is ready, and shuts it down when the job ends. The service log is at `logs/engine.log` in the run directory. The generated config sets `repeat_penalty=1.1` to avoid repetition on long text.

There is no `--ocr-dir` above, on purpose: llama.cpp and Ollama load the GGUF files directly and never touch the HuggingFace-format model directory. `--max-model-len` is 8192 here rather than the 40960 used for SGLang / vLLM because the GGUF backends allocate KV cache for the whole context up front, and a large value costs a lot of VRAM — while OCRCPM recognizes one layout block at a time, for which 8192 is plenty.

Layout detection still runs on CPU inside OCRCPM; only the OCR model uses llama.cpp.

## 7. Ollama backend (optional)

Ollama has upgraded its bundled llama.cpp to a version that includes the MiniCPM-V 4.x adaptation, so the GGUF pair from section 6 can be reused directly without building llama.cpp.

Ollama publishes prebuilt binaries, so there is no source and nothing to compile on the target machine. Version 0.32.6-rc0 or newer is required; this project was fully verified on that version. First check that `ollama` is available:

```bash
ollama --version
```

When the service is not running, this prints `ollama version is 0.0.0` plus a line `Warning: client version is 0.32.6-rc0`. That is normal — the second one is the real version.

If it is not installed, download and unpack the official release (x86_64 with an NVIDIA GPU uses `ollama-linux-amd64.tar.zst`, about 1.4 GB):

```bash
curl -fL -o ollama-linux-amd64.tar.zst \
  https://github.com/ollama/ollama/releases/download/v0.32.6-rc0/ollama-linux-amd64.tar.zst
mkdir -p ollama-root
tar --use-compress-program=unzstd -xf ollama-linux-amd64.tar.zst -C ollama-root
```

Only zstd archives are published, so install zstd first if `unzstd` is missing (`apt install zstd` or `yum install zstd`). Packages for ARM64, ROCm and other targets, plus newer versions, are at <https://github.com/ollama/ollama/releases>. The unpacked tree is about 2.1 GB:

```text
ollama-root/
├── bin/ollama                 # main binary
└── lib/ollama/                # ggml runtime, with both cuda_v12 and cuda_v13 backends
```

The relative position of `bin` and `lib` must not change; the binary looks for `../lib/ollama`. It does not need to be installed into a system directory — pass `--ollama-bin /abs/path/to/ollama-root/bin/ollama` during init. If disk is tight, keep only the CUDA backend matching your driver (`cuda_v13` for driver 13.x, `cuda_v12` for 12.x) to save about 1 GB.

Write the Ollama backend config:

```bash
ocrcpm init \
  --skip-download \
  --backend ollama \
  --layout-dir "$PWD/layout_model" \
  --ollama-model-gguf "$PWD/ocr_gguf/model-f16.gguf" \
  --ollama-mmproj "$PWD/ocr_gguf/mmproj-f16.gguf" \
  --model-name minicpm-v-ocr \
  --layout-device cpu \
  --gpu-id 0 \
  --port 11434 \
  --max-model-len 8192 \
  --input "$PWD/demo.png"
```

Recognition still uses the same command:

```bash
ocrcpm parse --input "$PWD/demo.png" --output "$PWD/demo.md"
```

On the first run OCRCPM starts `ollama serve`, runs `ollama create` once with the two GGUF files, waits for the model to be ready, then starts recognizing, and stops the service when the job ends. Registration happens only the first time; later runs reuse the registered model. Both the service log and the `ollama create` output go to `logs/engine.log` in the run directory.

Note that `ollama create` copies the GGUF into Ollama's own store (`~/.ollama/models` by default). During registration the language model is stored as-is and then stored again with rewritten metadata, and the old copy is not cleaned up — a 2.8 GB GGUF was measured to occupy about 4.7 GB. Together with the original files, reserve roughly 8 GB. If that partition is short on space, add `--ollama-models-dir /abs/path/to/ollama_models` during init.

If the model is already registered in Ollama, omit the two GGUF parameters and OCRCPM will use whatever `--model-name` points at. To register manually, the two `FROM` lines point at the language model and the projector; Ollama figures out which one is the projector:

```text
FROM /abs/path/to/ocr_gguf/model-f16.gguf
FROM /abs/path/to/ocr_gguf/mmproj-f16.gguf
```

```bash
ollama create minicpm-v-ocr -f Modelfile
```

OCRCPM talks to Ollama's native `/api/chat` endpoint rather than `/v1/chat/completions`, because Ollama's OpenAI-compatible endpoint does not support `top_k` and `repeat_penalty`, which OCR relies on for greedy decoding and repetition suppression.

This command needs no `--ocr-dir` either, for the same reason as section 6. `--port 11434` is Ollama's default; change it if another Ollama service already holds that port.

## 8. SGLang backend (optional)

Do not use the official build from `pip install sglang`. The official repository and PyPI currently lack this OCR model's registration and config parsing, so the official build fails while loading the model. Use the branch carrying the adaptation, [`support-minicpm-ocr` of tc-mb/sglang](https://github.com/tc-mb/sglang/tree/support-minicpm-ocr); it only adds model class, config and multimodal processor registration on top of the official code, with no change to the inference kernels.

```bash
git clone --branch support-minicpm-ocr https://github.com/tc-mb/sglang.git "$PWD/sglang"
git -C "$PWD/sglang" checkout 775f12f0d
```

`775f12f0d` is the commit verified by this project. Create a separate environment and install it editable (Python 3.10+, verified on 3.12):

```bash
python -m venv "$PWD/sglang_env"
source "$PWD/sglang_env/bin/activate"
python -m pip install -U pip
python -m pip install -e "$PWD/sglang/python"
deactivate
```

SGLang is pure Python and needs no compilation, but it pulls torch, flashinfer and friends, which can take 15 minutes depending on the network. Afterwards confirm that this very source tree is what got installed — the output should contain `Editable project location` pointing at the cloned `sglang/python`:

```bash
"$PWD/sglang_env/bin/python" -m pip show sglang | grep -E "Version|Editable"
```

Because it is an editable install, the `sglang/` directory must not be deleted or moved after deployment, or the environment breaks.

Write the SGLang backend config:

```bash
ocrcpm init \
  --skip-download \
  --backend sglang \
  --layout-dir "$PWD/layout_model" \
  --ocr-dir "$PWD/ocr_model" \
  --sglang-python "$PWD/sglang_env/bin/python" \
  --cuda-home /usr/local/cuda \
  --model-name minicpm-v-ocr \
  --layout-device cpu \
  --gpu-id 0 \
  --port 19699 \
  --max-model-len 40960 \
  --max-num-batched-tokens 40960 \
  --gpu-memory-utilization 0.8 \
  --input "$PWD/demo.png"
```

`--cuda-home` must point at a CUDA toolchain directory containing `bin/nvcc` (if nvcc lives in a conda environment, use that environment's path). With it set, OCRCPM adds the matching CUDA runtime to the dynamic library search path and isolates the TVM-FFI JIT cache per toolchain, so a cache produced by a different CUDA version is not reused.

Then run:

```bash
ocrcpm parse --input "$PWD/demo.png" --output "$PWD/demo.md"
```

OCRCPM starts SGLang, waits for `/v1/models`, and shuts it down when the job ends. `OCRCPM_SGLANG_PYTHON` can supply the environment's Python instead. The service log goes to `logs/engine.log` in the run directory, which is also where you confirm that the OCR model implementation was selected rather than a fallback.

On a 24 GB card, `--max-model-len 40960` with the default `gpu_memory_utilization` starts fine and needs no tuning.

## 9. vLLM backend (optional)

Official vLLM cannot load this model either. Use the branch carrying the adaptation, [`tc-ocr` of tc-mb/vllm](https://github.com/tc-mb/vllm/tree/tc-ocr), which dispatches the model implementation on the `version` field in the model directory's `config.json`.

```bash
git clone --branch tc-ocr https://github.com/tc-mb/vllm.git "$PWD/vllm"
git -C "$PWD/vllm" checkout 87fcfae14
```

`87fcfae14` is the commit verified by this project. Besides model support, the branch carries a server-side OCR pipeline mounted on `/v1/chat/completions`, but that part is only enabled when the server is started with `--ocr-layout-model`. OCRCPM does layout splitting on the client, does not pass that flag, and therefore treats the server as a plain inference service.

Create a separate environment and install editable. This step compiles CUDA extensions, so prepare a CUDA toolchain first (this commit requires torch 2.13, i.e. CUDA 13.0):

```bash
python -m venv "$PWD/vllm_env"
source "$PWD/vllm_env/bin/activate"
python -m pip install -U pip
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=64
python -m pip install -e "$PWD/vllm"
deactivate
```

Set `TORCH_CUDA_ARCH_LIST` to the target GPU's compute capability (`8.9` for RTX 4090, `8.0` for A100; query it with `nvidia-smi --query-gpu=compute_cap --format=csv`). Without it, every architecture is compiled, multiplying the build time. Tune `MAX_JOBS` to the core count.

Install time goes into two things: cmake fetches cutlass, triton, flash-attention and other third-party dependencies from GitHub (about 3 GB), then the CUDA kernels are compiled. On a 96-core machine restricted to a single architecture, compilation takes about 25 minutes; dependency downloads add anywhere from 15 minutes to an hour depending on the network.

`flashinfer` is installed automatically as a dependency. If startup reports `ModuleNotFoundError: No module named 'flashinfer'`, add it manually:

```bash
"$PWD/vllm_env/bin/python" -m pip install flashinfer-python nvidia-ml-py
```

As with SGLang this is an editable install, so the cloned `vllm/` directory must not be deleted or moved after deployment.

Write the vLLM backend config:

```bash
ocrcpm init \
  --skip-download \
  --backend vllm-auto \
  --layout-dir "$PWD/layout_model" \
  --ocr-dir "$PWD/ocr_model" \
  --vllm-python "$PWD/vllm_env/bin/python" \
  --cuda-home /usr/local/cuda \
  --model-name minicpm-v-ocr \
  --layout-device cpu \
  --gpu-id 0 \
  --port 19699 \
  --max-model-len 40960 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.88 \
  --input "$PWD/demo.png"
```

`--cuda-home` must point at a CUDA toolchain containing `bin/nvcc`: some flashinfer kernels are JIT-compiled at startup, and without nvcc you get `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`.

Then run:

```bash
ocrcpm parse --input "$PWD/demo.png" --output "$PWD/demo.md"
```

OCRCPM launches the service as `python -m vllm.entrypoints.openai.api_server`, automatically adding `--trust-remote-code`, `--enforce-eager` and the `--max-model-len`, `--limit-mm-per-prompt` and `--gpu-memory-utilization` derived from the config, waits for `/v1/models`, then starts recognizing and stops the service at the end. `OCRCPM_VLLM_PYTHON` can supply the environment's Python instead.

The `--max-num-batched-tokens 8192` and `--gpu-memory-utilization 0.88` above are a combination verified on a 24 GB card. Do not copy the 40960 from the SGLang section: vLLM pre-allocates VRAM for a profiling pass based on this value and 40960 will OOM right there, while `--max-model-len` can stay at 40960 without trouble. Cards with more VRAM can raise it for more throughput.

If a vLLM service already runs elsewhere, skip the auto-start path: set `engine.auto_start` to `false` in the config and put the service address in `engine.server_urls`. With several addresses listed, requests are distributed randomly with automatic failover.

## 10. Test with Open WebUI (optional)

The package ships two server entry points and the distinction matters: `ocrcpm-server` exposes the **crop-level** OCR model, so sending a full page to it bypasses layout detection and Markdown assembly; `ocrcpm-pipeline-server` is the one that exposes the whole pipeline as a single model. Document front-ends should use the latter.

First write a server-mode config:

```bash
ocrcpm init \
  --skip-download \
  --layout-dir "$PWD/layout_model" \
  --ocr-dir "$PWD/ocr_model" \
  --backend hf-server \
  --server-url "http://127.0.0.1:18599" \
  --layout-device cpu \
  --gpu-id 0 \
  --input "$PWD/demo.png" \
  --run-root "$PWD/runs" \
  --config-output "$PWD/ocrcpm_config.yaml"
```

Then start the OCR service and the pipeline service in separate terminals:

```bash
# terminal 1: OCR service
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
ocrcpm-server \
  --model-path "$PWD/ocr_model" \
  --host 127.0.0.1 \
  --port 18599
```

```bash
# terminal 2: pipeline service
ocrcpm-pipeline-server \
  --config "$PWD/ocrcpm_config.yaml" \
  --host 0.0.0.0 \
  --port 18600 \
  --run-root "$PWD/openwebui_runs"
```

Check both services; they should report the OCR model and `ocrcpm-pipeline` respectively:

```bash
curl http://127.0.0.1:18599/v1/models
curl http://127.0.0.1:18600/v1/models
```

Open WebUI is best kept in its own environment:

```bash
conda create -n open-webui python=3.11 -y
conda activate open-webui
python -m pip install open-webui

cd ~/ocrcpm_demo
DATA_DIR="$PWD/open-webui-data" \
ENABLE_OPENAI_API=True \
OPENAI_API_BASE_URL="http://127.0.0.1:18600/v1" \
OPENAI_API_KEY="EMPTY" \
ENABLE_OLLAMA_API=False \
WEBUI_AUTH=False \
ENABLE_PERSISTENT_CONFIG=False \
open-webui serve --host 0.0.0.0 --port 3000
```

If Open WebUI runs on a different machine than OCRCPM, replace `127.0.0.1` in `OPENAI_API_BASE_URL` with the OCRCPM server's IP. It must connect to the pipeline service on `18600/v1`, never directly to `18599/v1`.

Open `http://<open-webui-host>:3000`, start a new chat, pick `ocrcpm-pipeline`, upload the test image, type `OCR` and send. The result comes back as Markdown — both plain JSON and OpenAI-compatible SSE streaming are supported — and run files are kept in `~/ocrcpm_demo/openwebui_runs/`.

## 11. Troubleshooting

### Layout or OCR model not found

```text
layout.model_dir not found
engine.model_path not found
```

Check that `config.json` exists in the corresponding directory:

```bash
ls "$PWD/layout_model/config.json"
ls "$PWD/ocr_model/config.json"
```

### CUDA unavailable

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

If this prints `False`, install a PyTorch build matching the machine's CUDA. You can also switch to `--transformers-device cpu` during init, but it is very slow and memory hungry.

### llama-server will not start

Re-check CUDA, the CMake build and the service log:

```bash
nvidia-smi
cmake --build "$PWD/llama.cpp/build" --config Release -j
test -x "$PWD/llama.cpp/build/bin/llama-server"
```

If the port is taken, change `--port` during init. Detailed startup errors are in `logs/engine.log` of the run directory.

### The Ollama model did not register

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama list
```

The model named by `--model-name` should be listed. If not, check the `ollama create` output in `logs/engine.log` of the run directory, and confirm that both GGUF paths are correct and that the projector file really is the `mmproj` one.

If `ollama serve` fails to start, port 11434 is usually already held by an existing Ollama service. Either change `--port`, or reuse that service: set `engine.auto_start` to `false` and put its address in `engine.server_urls`.

### SGLang or vLLM reports an unsupported model architecture

Most likely the environment holds the official build rather than the adapted branch from section 8 or 9. The vLLM error looks like:

```text
ValueError: Currently, MiniCPMV only supports versions ...
```

Confirm which code is installed:

```bash
"$PWD/sglang_env/bin/python" -m pip show sglang | grep -E "Version|Editable"
"$PWD/vllm_env/bin/python" -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)"
```

SGLang should show `Editable project location` pointing at the cloned `sglang/python`; vLLM should point at the cloned `vllm/vllm` directory rather than a normal install under `site-packages`. The same symptom appears if `sglang/` or `vllm/` was moved or deleted after installation. Do not run the vLLM command from inside the `vllm/` directory, or `import vllm` resolves to the current directory and gives a misleading answer.

Both backends also require the `version` field in the model directory's `config.json` to match what the branch expects, since that is the field the adaptation dispatches on.

### Loading errors related to transformers 5.x

The glue code shipped with the weights has to match the installed transformers version. Two cases are known: if `processing_minicpmv.py` does not pass `downsample_mode` through, or does not call `get_slice_image_placeholder` with its new signature, the image placeholder count comes out as 0 and every block fails on the transformers backend; and reading `tokenizer.bos_id` in `modeling_minicpmv.py` raises `TokenizersBackend has no attribute bos_id` on transformers 5.x, so it has to fall back to `bos_token_id`.

Both only surface on the transformers backend, because vLLM and SGLang use their own model implementations and never reach those functions.

### A wrong config was generated, or a backend switch had no effect

Every `ocrcpm init` writes the same `~/.config/ocrcpm/config.yaml`, so re-running the command from the relevant section overwrites it. To see which backend is active, look at `engine.type` in that file:

| `engine.type` | Backend |
| --- | --- |
| `minicpm_transformers_local` | transformers-local (section 4) |
| `minicpm_llamacpp_openai_api` | llama.cpp (section 6) |
| `minicpm_ollama_api` | Ollama (section 7) |
| `minicpm_sglang_openai_api` | SGLang (section 8) |
| `minicpm_vllm_openai_api` | vLLM (section 9) or http-client mode |

The server-mode config from section 10 lives in a separate `ocrcpm_config.yaml` and does not affect the default config.

## 12. Python API

Besides the CLI, the pipeline can be driven stage by stage from Python:

```python
from ocrcpm import load_config, parse_layout_run, prepare_run, infer_run

cfg = load_config("my_config.yaml")
parse_layout_run(cfg)
prepare_run(cfg)
print(infer_run(cfg))
```
