# MiniCPM-o 4.5 · llama.cpp-omni Usage Guide

This document describes how to run multimodal inference for MiniCPM-o 4.5 on top of the **llama.cpp-omni** repository, covering three main usages:

1. **Command-line CLI** (`tools/omni/omni-cli.cpp` → `llama-omni-cli`)
2. **Test programs** (`tools/omni/test/` → simplex audio / simplex omni / duplex)
3. **Omni server** (`tools/server/ws_handler.cpp` etc. → `llama-omni-server`)

Plus the accompanying **Demo** (HuggingFace Space and the locally-deployable MiniCPM-o-Demo).

> The examples in this document are verified on Linux + NVIDIA CUDA (RTX 4090); on macOS replace `CUDA` with `Metal`, the commands are the same.

---

## Table of Contents

- [Part 1: llama.cpp-omni](#part-1-llamacpp-omni)
  - [1. Environment & Build](#1-environment--build)
  - [2. Download the Model](#2-download-the-model)
  - [3. Usage 1: CLI (`llama-omni-cli`)](#3-usage-1-cli-llama-omni-cli)
  - [4. Usage 2: Test Programs (`tools/omni/test/`)](#4-usage-2-test-programs-toolsomnitest)
  - [5. Usage 3: Omni Server (`llama-omni-server`)](#5-usage-3-omni-server-llama-omni-server)
- [Part 2: Demo](#part-2-demo)

---

# Part 1: llama.cpp-omni

llama.cpp-omni splits MiniCPM-o 4.5 into several independent GGUF modules that cooperate during inference:

| Module | Role |
| --- | --- |
| **LLM** | Main language model (Qwen3-8B); receives vision/audio embeddings and generates text tokens |
| **VPM** (vision) | Vision encoder (SigLip2 + Resampler); encodes images into the LLM hidden space |
| **APM** (audio) | Audio encoder (Whisper); encodes 16 kHz audio into the LLM hidden space |
| **TTS** | Text-to-speech model; turns LLM hidden states into audio tokens |
| **Token2Wav** | Vocoder (Flow Matching + HiFiGAN); synthesizes audio tokens into 24 kHz waveforms |
| **projector** | Projection layer used by TTS |

---

## 1. Environment & Build

### 1.1 Dependencies

- CMake 3.14+ and a C++17 compiler
- GPU backend (auto-detected): Linux + NVIDIA → CUDA; macOS → Metal
- (Optional) OpenSSL: llama.cpp-omni defaults to `LLAMA_OPENSSL=ON`, used for serving HTTPS. This option directly affects how `llama-omni-server` starts, see [5.1](#51-starting-the-server-important)
- (For server video input) `ffmpeg`: needed to parse uploaded MP4s in `turn_based` mode

### 1.2 Build

```bash
# Configure (CMake will auto-detect and enable CUDA / Metal)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build the targets you need
cmake --build build --config Release -j \
    --target llama-omni-cli \
             llama-omni-single-test-audio \
             llama-omni-single-test-omni \
             llama-omni-test-duplex \
             llama-omni-server
```

Artifacts are placed in `build/bin/`:

| Binary | Description |
| --- | --- |
| `llama-omni-cli` | CLI inference tool (Usage 1 in this doc) |
| `llama-omni-single-test-audio` | Simplex · audio-only batch test |
| `llama-omni-single-test-omni` | Simplex · audio + vision batch test |
| `llama-omni-test-duplex` | Full-duplex test |
| `llama-omni-server` | Omni HTTP/WebSocket server (Usage 3 in this doc) |

> If you do not want HTTPS, add `-DLLAMA_OPENSSL=OFF` at configure time; `llama-omni-server` will then start with plain HTTP (no certificate needed).

---

## 2. Download the Model

### 2.1 Directory layout (must be consistent)

The CLI / tests / server only need the path to the **LLM** (`-m`); the other sub-models are auto-inferred from the following **fixed directory layout**, which must be kept consistent:

```
MiniCPM-o-4_5-gguf/
├── MiniCPM-o-4_5-Q4_K_M.gguf         # LLM; can be swapped for F16 / Q8_0 / Q4_K_M etc.
├── audio/
│   └── MiniCPM-o-4_5-audio-F16.gguf
├── vision/
│   └── MiniCPM-o-4_5-vision-F16.gguf
├── tts/
│   ├── MiniCPM-o-4_5-tts-F16.gguf
│   └── MiniCPM-o-4_5-projector-F16.gguf
└── token2wav-gguf/                   # required when TTS is enabled
    ├── encoder.gguf                  # ~144MB
    ├── flow_matching.gguf            # ~437MB
    ├── flow_extra.gguf               # ~13MB
    ├── hifigan2.gguf                 # ~79MB
    └── prompt_cache.gguf             # ~67MB
```

Notes:
- **Pick any single LLM quantization**; you do not need to download all of them. VRAM/RAM reference: Q4_K_M ~9GB, Q8_0 ~13GB, F16 ~20GB (Full Omni).
- The `audio/`, `vision/`, and `tts/` sub-directories are recommended to be downloaded together; `token2wav-gguf/` can be omitted only when using `--no-tts`.
- If you only use a single modality (vision or audio), you may download only the corresponding sub-model, but keep the directory names unchanged.

### 2.2 Option 1: Download official GGUF directly (recommended)

- HuggingFace: <https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf>
- ModelScope: <https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf>

```bash
# HuggingFace (huggingface-cli)
pip install -U "huggingface_hub[cli]"
huggingface-cli download openbmb/MiniCPM-o-4_5-gguf \
    --local-dir ./MiniCPM-o-4_5-gguf

# Or ModelScope
pip install -U modelscope
modelscope download --model OpenBMB/MiniCPM-o-4_5-gguf \
    --local_dir ./MiniCPM-o-4_5-gguf
```

After downloading, confirm the directory layout matches [2.1](#21-directory-layout-must-be-consistent) above (especially `token2wav-gguf/` and `tts/projector`).

### 2.3 Option 2: Convert from PyTorch weights yourself

First download the original weights (<https://huggingface.co/openbmb/MiniCPM-o-4_5>), then convert with the repository script:

```bash
# Edit the paths at the top of the script, then run
#   MODEL_DIR   = PyTorch model directory
#   OUTPUT_DIR  = output gguf directory
bash ./tools/omni/convert/run_convert.sh
```

The script sequentially performs surgery (component splitting) → converts VPM/APM/LLM/TTS/projector → quantizes the LLM. Token2Wav must be converted separately following the instructions in `tools/omni/token2wav/`.

---

## 3. Usage 1: CLI (`llama-omni-cli`)

The CLI is the most direct command-line entry point; pass the LLM path via `-m` and it will run a built-in test case and write the synthesized speech to `tools/omni/output/`.

> It is recommended to run from the repository root, since the default reference audio and test data use relative paths (`tools/omni/assets/...`). For single-GPU runs you may add `CUDA_VISIBLE_DEVICES=0`.

```bash
export MODEL=/path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf

# Basic usage (runs the built-in audio-only case by default; other sub-model paths are auto-inferred)
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-cli -m "$MODEL"

# Specify an audio test case: <prefix> <count>, files named like <prefix>0000.wav
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-cli -m "$MODEL" \
    --test tools/omni/assets/test_case/audio_test_case/audio_test_case_ 2

# Omni mode (audio + vision, auto-matches a .jpg with the same name)
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-cli -m "$MODEL" --omni \
    --test tools/omni/assets/test_case/omni_test_case/omni_test_case_ 9

# Text-only output (disable TTS, no speech generated)
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-cli -m "$MODEL" --no-tts
```

### Common Arguments

| Argument | Description |
| --- | --- |
| `-m <path>` | **Required**; path to the LLM GGUF (other sub-models are inferred from it) |
| `--vision/--audio/--tts/--projector <path>` | Override the corresponding sub-model path |
| `--ref-audio <path>` | Reference audio for voice cloning (default `tools/omni/assets/default_ref_audio/default_ref_audio.wav`) |
| `-c, --ctx-size <n>` | Context length (default 4096) |
| `-ngl <n>` | Number of GPU layers (default 99, i.e. offload as much as possible to GPU) |
| `--no-tts` | Disable TTS, output text only |
| `--omni` | Enable omni (audio + vision) |
| `--test <prefix> <n>` | Specify the test data prefix and count |

For the full list of arguments see `./build/bin/llama-omni-cli -h`.

### Output

Synthesized speech is written per round to `tools/omni/output/round_XXX/tts_wav/wav_*.wav`; text and debug info are in the sibling `llm_debug/` directory. Use `tools/omni/test/merge_wav.sh` to merge the shards into a single file.

---

## 4. Usage 2: Test Programs (`tools/omni/test/`)

The three test programs share the same model auto-inference logic and `--test <prefix> <n>` syntax; they can be used for functional verification, result alignment, and performance stress testing.

### 4.1 Simplex · audio-only `llama-omni-single-test-audio`

Handles audio only: all chunks are prefilled synchronously and then decoded once together.

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-single-test-audio -m "$MODEL" \
    --test tools/omni/assets/test_case/audio_test_case/audio_test_case_ 2
```

### 4.2 Simplex · audio + vision `llama-omni-single-test-omni`

Each chunk's `.wav` is paired with a `.jpg` of the same name (missing image degrades to audio-only).

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-single-test-omni -m "$MODEL" \
    --test tools/omni/assets/test_case/omni_test_case/omni_test_case_ 5
```

### 4.3 Duplex `llama-omni-test-duplex`

Simulates full-duplex streaming input: frames are pushed at a fixed pace, and the model outputs a `<|speak|>` or `<|listen|>` token per frame.

```bash
# --stream-interval 1000 means one frame per second (simulating real streaming); 0 means continuous stress test with no delay
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-test-duplex -m "$MODEL" --omni \
    --test tools/omni/assets/test_case/omni_test_case/omni_test_case_ 5 \
    --stream-interval 1000
```

At the end it prints the per-frame decode/e2e timing and a speak/listen summary.

> Test data lives in `tools/omni/assets/test_case/` (including `audio_test_case/`, `omni_test_case/`, etc.); files are named `<prefix>0000.wav`, `<prefix>0001.wav` and so on, and the argument `<n>` is the number of chunks to read.

---

## 5. Usage 3: Omni Server (`llama-omni-server`)

`llama-omni-server` provides two sets of interfaces; the server-side logic is mainly in `tools/server/ws_handler.cpp`:

- **WebSocket `/backend`**: modern protocol (session.init → input.append → event stream), which the Demo is built upon; supports both `turn_based` (turn-by-turn dialog) and `full_duplex` (full-duplex streaming) modes.
- **Legacy HTTP (SSE)**: `/v1/stream/omni_init`, `/v1/stream/prefill`, `/v1/stream/decode`, etc., convenient for manual debugging with curl.
- **Health checks**: `GET /health`, `GET /v1/health`.

### 5.1 Starting the server (important)

> llama.cpp-omni defaults to `LLAMA_OPENSSL=ON`, and the server starts with HTTPS (SSLServer). If no certificate is provided, the process exits immediately after printing `Omni HTTP server starting...`, and the port is not listened on. Choose one of the two approaches below:

**Option A: Provide a certificate and start with HTTPS (required by the Demo / browsers)**

```bash
# Generate a self-signed certificate (for local testing)
mkdir -p /tmp/omni_ssl
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
    -keyout /tmp/omni_ssl/key.pem -out /tmp/omni_ssl/cert.pem \
    -subj "/CN=localhost"

# Start (-m points to the LLM; other sub-models are auto-inferred)
export MODEL=/path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-server \
    -m "$MODEL" --host 0.0.0.0 --port 28099 -c 4096 -ngl 99 \
    --ssl-cert-file /tmp/omni_ssl/cert.pem \
    --ssl-key-file  /tmp/omni_ssl/key.pem
```

**Option B: Disable OpenSSL at configure time and start with plain HTTP**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF
cmake --build build --config Release -j --target llama-omni-server
# Afterwards, no --ssl-* arguments are needed; access via http://
```

Other notes:
- **Port in use**: if the port is already occupied by another process, the server will fail to bind and exit. Switch to a free port or release the occupied one first.
- **Lazy model loading**: the server itself starts quickly, but the model is only loaded on the first WebSocket session / first `omni_init` call, so the first call takes 10–60s to load.

Verify a successful start:

```bash
# For HTTPS (Option A) use -k to skip self-signed certificate verification
curl -sk https://127.0.0.1:28099/health
# → {"engine":"comni","status":"ok"}
```

### 5.2 Calling via WebSocket `/backend` (modern protocol / used by the Demo)

Message format:

- Upstream first frame `session.init` (`payload.mode` = `turn_based` or `full_duplex`; `payload.use_tts` controls whether audio is emitted)
- Upstream `input.append` (`turn_based` uses `input.messages`; `full_duplex` uses `input.audio` + optional `input.video_frames`)
- Downstream events: `session.created` → `response.output.delta` (`kind` = `text`/`audio`/`listen`) → `response.done` → `session.closed`

A minimal `turn_based` text-only example (Python, `pip install websockets`):

```python
import asyncio, json, ssl, websockets

async def main():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # for self-signed certs; in HTTP mode use ws:// and drop the ssl arg
    async with websockets.connect("wss://127.0.0.1:28099/backend", ssl=ctx, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "turn_based", "use_tts": False}
        }))
        print(json.loads(await ws.recv())["type"])   # session.created

        await ws.send(json.dumps({
            "type": "input.append",
            "input": {
                "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
                "streaming": True,
                "generation": {"max_new_tokens": 128}
            }
        }))

        text = ""
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "response.output.delta" and m.get("kind") == "text":
                text += m.get("text", "")
            elif m["type"] == "response.done":
                print("Reply:", m.get("text")); break

asyncio.run(main())
```

- In `turn_based`, you can include `image`/`audio`/`video` content items in `messages[].content` for multimodal Q&A; when `input.use_tts_template=true`, the reply carries synthesized speech (`response.done.audio`, base64 float32 PCM).
- In `full_duplex`, each `input.append` sends 1 second of audio (+ optional video frames); the server decides speak/listen at 1 Hz for real-time conversation.

### 5.3 Calling via Legacy HTTP (SSE) (manual debugging with curl)

Call order: `omni_init` → loop (`prefill` → `decode`). `decode` returns an SSE text stream, and the synthesized speech is written under `output_dir`.

```bash
export B=https://127.0.0.1:28099    # In HTTP mode change to http:// and drop -k

# 1) Initialize (loads the model on first call; the system prompt prefill for index=0 is done internally)
curl -sk -X POST $B/v1/stream/omni_init -H 'Content-Type: application/json' \
    -d '{"media_type":1,"use_tts":true,"output_dir":"./tools/omni/output_server"}'

# 2) Send a user audio segment (audio_path_prefix is a server-visible file path; cnt is the index, starting from 1)
curl -sk -X POST $B/v1/stream/prefill -H 'Content-Type: application/json' \
    -d '{"audio_path_prefix":"tools/omni/assets/test_case/audio_test_case/audio_test_case_0000.wav","cnt":1}'

# 3) Trigger generation (SSE stream)
curl -sk -N -X POST $B/v1/stream/decode -H 'Content-Type: application/json' \
    -d '{"stream":true,"debug_dir":"./tools/omni/output_server"}'
```

The SSE response looks like (note the text field is **`content`**, not `text`):

```
data: {"content":"Once upon a time there was a mountain","end_of_turn":false,"is_listen":false,"stop":false}
data: {"content":"","end_of_turn":true,"is_listen":false,"stop":true}
data: [DONE]
```

| Field | Description |
| --- | --- |
| `content` | Text fragment (may be empty; filter before display) |
| `is_listen` | `true` means the model has switched to the listening state |
| `stop` | `true` means this turn of generation has ended |

> `media_type`: `1` = audio only, `2` = audio + vision; `cnt=0` of `prefill` is already handled internally by `omni_init`, so your own count starts from `1`.

---

# Part 2: Demo

The MiniCPM-o 4.5 Demo is a **full-duplex real-time audio/video conversation** web app, available both as an online experience and for local deployment.

## 1. HuggingFace Space (online experience)

Open the official hosted Demo for an online experience: <https://huggingface.co/spaces/openbmb/MiniCPM-o-4_5-Demo> (or visit <https://minicpmo45.modelbest.cn/> directly).

## 2. Local Deployment: MiniCPM-o-Demo (with llama.cpp-omni's C++ backend)

To run the full audio/video call frontend on your own machine backed by your self-built `llama-omni-server`, use the official Demo repository:

- Repository: <https://github.com/OpenBMB/MiniCPM-o-Demo> (use the default `main` branch)

> Branch note: the C++ backend (llama.cpp-omni) has been merged into `main`, which supports both the PyTorch backend and the C++ backend. The old `Comni` branch is deprecated; use `main` instead.

The Demo uses a gateway + worker + backend architecture, where worker/gateway is a generic Python orchestration layer, and the backend can be either PyTorch (`py_backend/server.py`) or C++ (llama.cpp-omni's `llama-omni-server`):

```
Browser ⇄ gateway.py (HTTPS, :8006) ⇄ worker.py (:22400) ⇄ llama-omni-server (HTTP, :22500)
```

### Option 1: Docker Compose (officially recommended)

The Demo provides `docker-compose.cpp.yml` for the C++ backend, which starts gateway + cpp-worker-backend with a single command. The cpp-worker-backend image automatically clones and compiles `tc-mb/llama.cpp-omni` inside the container, so no local build is required.

```bash
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git
cd MiniCPM-o-Demo

# Generate a self-signed certificate (for gateway HTTPS)
mkdir -p certs data
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
    -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"

# Build and start (GGUF_MODEL_HOST_PATH points to the host GGUF directory, mounted read-only)
GGUF_MODEL_HOST_PATH=/path/to/MiniCPM-o-4_5-gguf \
GATEWAY_HOST_PORT=8006 \
CPP_GPU_ID=0 \
docker compose -f docker-compose.cpp.yml up -d --build

docker compose -f docker-compose.cpp.yml logs -f cpp-worker-backend   # watch backend loading
# Open https://<host>:8006/ in a browser
```

Docker notes:
- **Docker Hub access required**: the image is based on `docker/dockerfile:1` and `nvidia/cuda:*` base images; if the machine cannot reach `docker.io` (common in offline / China environments), the build will time out while pulling base images. Configure a registry mirror or fall back to Option 2.
- **CUDA architecture**: `docker/Dockerfile.cpp-worker-backend` defaults to `CMAKE_CUDA_ARCHITECTURES=80;120` (A100 / Blackwell) and does not include Ada (sm_89, e.g. RTX 40 series). If you get `no kernel image is available for execution on the device` at runtime, add `89` to the arg in that Dockerfile and rebuild.
- Certificates are only used at the **gateway** layer (HTTPS terminates at the gateway); the `llama-omni-server` inside the container is plain HTTP and needs no certificate (see below).

### Option 2: Bare-metal (no Docker)

Suitable for debugging or environments without Docker Hub access. Start the three processes in order:

```bash
# 0) First build llama.cpp-omni's llama-omni-server (see Part 1, Section 1)
#    The backend must be the plain HTTP version: the demo's worker connects to the backend via ws://
#    and does not skip self-signed certificate verification, so it cannot reach an HTTPS backend.
#    Therefore, disable OpenSSL at build time:
#       cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF
#       cmake --build build --config Release -j --target llama-omni-server

git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git
cd MiniCPM-o-Demo

# 1) Install worker/gateway dependencies (C++ backend mode does not need torch; lightweight)
python3 -m venv .venv-cpp && . .venv-cpp/bin/activate
pip install "fastapi>=0.128.0" "httpx>=0.28.0" "numpy>=2.2.0" "pydantic>=2.11.0" \
            python-multipart "uvicorn>=0.40.0" "websockets>=16.0"

export MODEL=/path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf

# 2) Start the backend llama-omni-server (HTTP, :22500)
CUDA_VISIBLE_DEVICES=0 /path/to/llama.cpp-omni/build/bin/llama-omni-server \
    -m "$MODEL" --host 127.0.0.1 --port 22500 -c 8192 -ngl 99 &

# 3) Start the worker (remote backend mode, pointing at the backend)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python worker.py \
    --host 0.0.0.0 --port 22400 --gpu-id 0 \
    --backend-server-url http://127.0.0.1:22500 &

# 4) Start the gateway (use --http for local debugging; switch back to default HTTPS for browser camera/mic)
python gateway.py --host 0.0.0.0 --port 8006 --http --workers localhost:22400 &

# 5) Verify
curl http://127.0.0.1:22500/health   # backend: {"engine":"comni","status":"ok"}
curl http://127.0.0.1:22400/health   # worker: {"status":"healthy","worker_status":"idle",...}
curl http://127.0.0.1:8006/health    # gateway: {"status":"healthy",...}
# Open http://<host>:8006/ in a browser
```

Notes:
- The backend must be the plain HTTP version (`-DLLAMA_OPENSSL=OFF`). This also explains why the official Docker image does not install `libssl-dev` — precisely so that `llama-omni-server` is compiled as the HTTP version.
- **Port conflicts**: the default ports for worker, backend, and gateway (22400 / 22500 / 8006) may conflict on shared machines; the process silently exits when a port is taken, so use free ports instead.
- **Camera/mic require HTTPS**: when gateway runs with `--http`, the browser disables media devices and only text input works; for real use, drop `--http` (defaults to HTTPS, requires `certs/cert.pem`, `certs/key.pem`).
- In C++ backend mode, the worker does not load any PyTorch model; it just forwards `session.init / input.append` to the backend. The exposed WS routes are `/v1/worker/chat` (turn_based) and `/v1/worker/duplex` (full_duplex).

For more details see the Demo repository's `README.md` / `README_zh.md` (`main` branch) and `docker-compose.cpp.yml`, `docker/entrypoint-cpp-worker-backend.sh`.

---

## References

- llama.cpp-omni repository: <https://github.com/tc-mb/llama.cpp-omni>
- Model (PyTorch): <https://huggingface.co/openbmb/MiniCPM-o-4_5>
- Model (GGUF): <https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf> · <https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf>
- Online Demo: <https://minicpmo45.modelbest.cn/>
- Demo repository: <https://github.com/OpenBMB/MiniCPM-o-Demo> (`main` branch)
- HF Space: <https://huggingface.co/spaces/openbmb/MiniCPM-o-4_5-Demo>
