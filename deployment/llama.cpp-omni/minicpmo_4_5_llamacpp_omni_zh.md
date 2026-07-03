# MiniCPM-o 4.5 · llama.cpp-omni 使用文档

本文档介绍如何基于 **llama.cpp-omni** 仓库运行 MiniCPM-o 4.5 的多模态推理，涵盖三种主要用法：

1. **命令行 CLI**（`tools/omni/omni-cli.cpp` → `llama-omni-cli`）
2. **测试程序**（`tools/omni/test/` → 单工音频 / 单工 omni / 双工）
3. **Omni 服务**（`tools/server/ws_handler.cpp` 等 → `llama-omni-server`）

以及配套的 **Demo**（HuggingFace Space 与可本地部署的 MiniCPM-o-Demo）。

> 本文档示例在 Linux + NVIDIA CUDA（RTX 4090）环境下验证；macOS 将 `CUDA` 替换为 `Metal`，命令相同。

---

## 目录

- [第一部分：llama.cpp-omni](#第一部分llamacpp-omni)
  - [1. 环境与构建](#1-环境与构建)
  - [2. 下载模型](#2-下载模型)
  - [3. 用法一：CLI](#3-用法一cli-llama-omni-cli)
  - [4. 用法二：测试程序](#4-用法二测试程序-toolsomnitest)
  - [5. 用法三：Omni 服务](#5-用法三omni-服务-llama-omni-server)
- [第二部分：Demo](#第二部分demo)

---

# 第一部分：llama.cpp-omni

llama.cpp-omni 把 MiniCPM-o 4.5 拆成多个独立的 GGUF 模块协同推理：

| 模块 | 作用 |
| --- | --- |
| **LLM** | 主语言模型（Qwen3-8B），接收视觉/音频 embedding 生成文本 token |
| **VPM** (vision) | 视觉编码器（SigLip2 + Resampler），把图像编码进 LLM 隐空间 |
| **APM** (audio) | 音频编码器（Whisper），把 16kHz 音频编码进 LLM 隐空间 |
| **TTS** | 文本转语音模型，把 LLM 隐状态生成音频 token |
| **Token2Wav** | 声码器（Flow Matching + HiFiGAN），把音频 token 合成 24kHz 波形 |
| **projector** | TTS 使用的投影层 |

---

## 1. 环境与构建

### 1.1 依赖

- CMake 3.14 及以上、C++17 编译器
- GPU 后端（自动检测）：Linux + NVIDIA → CUDA；macOS → Metal
- （可选）OpenSSL：llama.cpp-omni 默认 `LLAMA_OPENSSL=ON`，用于服务 HTTPS。该选项直接影响 `llama-omni-server` 的启动方式，见 [5.1](#51-启动服务重点)
- （服务的视频输入）`ffmpeg`：`turn_based` 模式下解析上传的 MP4 需要

### 1.2 构建

```bash
# 配置（CMake 会自动探测并启用 CUDA / Metal）
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 构建需要用到的目标
cmake --build build --config Release -j \
    --target llama-omni-cli \
             llama-omni-single-test-audio \
             llama-omni-single-test-omni \
             llama-omni-test-duplex \
             llama-omni-server
```

产物位于 `build/bin/`：

| 二进制 | 说明 |
| --- | --- |
| `llama-omni-cli` | 命令行推理工具（本文档用法一） |
| `llama-omni-single-test-audio` | 单工·纯音频批量测试 |
| `llama-omni-single-test-omni` | 单工·音频+视觉批量测试 |
| `llama-omni-test-duplex` | 双工（full-duplex）测试 |
| `llama-omni-server` | Omni HTTP/WebSocket 服务（本文档用法三） |

> 若不想启用 HTTPS，配置时加 `-DLLAMA_OPENSSL=OFF`，`llama-omni-server` 就会用普通 HTTP 启动（无需证书）。

---

## 2. 下载模型

### 2.1 目录结构（必须一致）

CLI / 测试 / 服务都只需要传 **LLM 的路径**（`-m`），其它子模型会按下面的**固定目录结构**自动推断，需保持一致：

```
MiniCPM-o-4_5-gguf/
├── MiniCPM-o-4_5-Q4_K_M.gguf         # LLM，可换 F16 / Q8_0 / Q4_K_M 等量化
├── audio/
│   └── MiniCPM-o-4_5-audio-F16.gguf
├── vision/
│   └── MiniCPM-o-4_5-vision-F16.gguf
├── tts/
│   ├── MiniCPM-o-4_5-tts-F16.gguf
│   └── MiniCPM-o-4_5-projector-F16.gguf
└── token2wav-gguf/                   # 开启 TTS 时必需
    ├── encoder.gguf                  # ~144MB
    ├── flow_matching.gguf            # ~437MB
    ├── flow_extra.gguf               # ~13MB
    ├── hifigan2.gguf                 # ~79MB
    └── prompt_cache.gguf             # ~67MB
```

说明：
- **LLM 量化任选一种**即可，无需全部下载。显存/内存参考：Q4_K_M 约 9GB、Q8_0 约 13GB、F16 约 20GB（Full Omni）。
- `audio/`、`vision/`、`tts/` 三个子目录建议一并下载；仅使用 `--no-tts` 时可省略 `token2wav-gguf/`。
- 若只使用视觉或音频单一模态，可只下载对应子模型，但请保持目录名称不变。

### 2.2 方式一：直接下载官方 GGUF（推荐）

- HuggingFace：<https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf>
- ModelScope：<https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf>

```bash
# HuggingFace（huggingface-cli）
pip install -U "huggingface_hub[cli]"
huggingface-cli download openbmb/MiniCPM-o-4_5-gguf \
    --local-dir ./MiniCPM-o-4_5-gguf

# 或 ModelScope
pip install -U modelscope
modelscope download --model OpenBMB/MiniCPM-o-4_5-gguf \
    --local_dir ./MiniCPM-o-4_5-gguf
```

下载后确认目录结构与上面 [2.1](#21-目录结构必须一致) 一致（尤其是 `token2wav-gguf/` 和 `tts/projector`）。

### 2.3 方式二：从 PyTorch 权重自行转换

先下载原始权重（<https://huggingface.co/openbmb/MiniCPM-o-4_5>），再用仓库脚本转换：

```bash
# 编辑脚本顶部路径后运行
#   MODEL_DIR   = PyTorch 模型目录
#   OUTPUT_DIR  = 输出 gguf 目录
bash ./tools/omni/convert/run_convert.sh
```

脚本会依次做 surgery（拆分组件）→ 转换 VPM/APM/LLM/TTS/projector → 量化 LLM。Token2Wav 需按 `tools/omni/token2wav/` 说明单独转换。

---

## 3. 用法一：CLI (`llama-omni-cli`)

CLI 是最直接的命令行入口，`-m` 传 LLM 路径即可，会运行一段内置测试用例并把合成语音写到 `tools/omni/output/`。

> 建议在仓库根目录运行，默认参考音频、测试数据均为相对路径（`tools/omni/assets/...`）。单卡运行可加 `CUDA_VISIBLE_DEVICES=0`。

```bash
export MODEL=/path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf

# 基本用法（默认运行内置纯音频用例，自动推断其它子模型路径）
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-cli -m "$MODEL"

# 指定音频测试用例：<前缀> <数量>，文件形如 <前缀>0000.wav
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-cli -m "$MODEL" \
    --test tools/omni/assets/test_case/audio_test_case/audio_test_case_ 2

# Omni 模式（音频 + 视觉，自动匹配同名 .jpg）
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-cli -m "$MODEL" --omni \
    --test tools/omni/assets/test_case/omni_test_case/omni_test_case_ 9

# 纯文本输出（关闭 TTS，不生成语音）
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-cli -m "$MODEL" --no-tts
```

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `-m <path>` | **必填**，LLM GGUF 路径（其它子模型据此推断） |
| `--vision/--audio/--tts/--projector <path>` | 覆盖对应子模型路径 |
| `--ref-audio <path>` | 声音克隆的参考音频（默认 `tools/omni/assets/default_ref_audio/default_ref_audio.wav`） |
| `-c, --ctx-size <n>` | 上下文长度（默认 4096） |
| `-ngl <n>` | GPU 层数（默认 99，即尽量全放 GPU） |
| `--no-tts` | 关闭 TTS，只输出文本 |
| `--omni` | 开启 omni（音频+视觉） |
| `--test <prefix> <n>` | 指定测试数据前缀与数量 |

完整参数见 `./build/bin/llama-omni-cli -h`。

### 输出

合成语音按轮次写到 `tools/omni/output/round_XXX/tts_wav/wav_*.wav`，文本与调试信息在同级 `llm_debug/`。可用 `tools/omni/test/merge_wav.sh` 把分片合并成整段。

---

## 4. 用法二：测试程序 (`tools/omni/test/`)

三个测试程序共用相同的模型自动推断逻辑和 `--test <prefix> <n>` 语法，可用于功能验证、结果对齐与性能压测。

### 4.1 单工·纯音频 `llama-omni-single-test-audio`

只处理音频：所有 chunk 同步 prefill 后统一 decode 一次。

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-single-test-audio -m "$MODEL" \
    --test tools/omni/assets/test_case/audio_test_case/audio_test_case_ 2
```

### 4.2 单工·音频+视觉 `llama-omni-single-test-omni`

每个 chunk 的 `.wav` 与同名 `.jpg` 配对（缺图则退化为纯音频）。

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-single-test-omni -m "$MODEL" \
    --test tools/omni/assets/test_case/omni_test_case/omni_test_case_ 5
```

### 4.3 双工 `llama-omni-test-duplex`

模拟 full-duplex 流式输入：按固定节奏推送帧，模型逐帧输出 `<|speak|>` 或 `<|listen|>` 标记。

```bash
# --stream-interval 1000 表示每 1 秒输入一帧（模拟真实流式）；0 表示无间隔连续压测
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-test-duplex -m "$MODEL" --omni \
    --test tools/omni/assets/test_case/omni_test_case/omni_test_case_ 5 \
    --stream-interval 1000
```

结束会打印每帧 decode/e2e 耗时与 speak/listen 汇总。

> 测试数据位于 `tools/omni/assets/test_case/`（含 `audio_test_case/`、`omni_test_case/` 等）；文件命名规则为 `<前缀>0000.wav`、`<前缀>0001.wav` 依次递增，参数 `<n>` 为要读取的 chunk 数量。

---

## 5. 用法三：Omni 服务 (`llama-omni-server`)

`llama-omni-server` 同时提供两套接口，服务端逻辑主要在 `tools/server/ws_handler.cpp`：

- **WebSocket `/backend`**：现代协议（session.init → input.append → 事件流），Demo 即基于此协议；支持 `turn_based`（按轮对话）与 `full_duplex`（全双工流式）两种模式。
- **Legacy HTTP（SSE）**：`/v1/stream/omni_init`、`/v1/stream/prefill`、`/v1/stream/decode` 等，便于使用 curl 手动调试。
- **健康检查**：`GET /health`、`GET /v1/health`。

### 5.1 启动服务（重点）

> llama.cpp-omni 默认 `LLAMA_OPENSSL=ON`，服务以 HTTPS（SSLServer）启动。若不提供证书，进程会在打印 `Omni HTTP server starting...` 后直接退出，端口不会监听。以下两种方式任选其一：

**方式 A：提供证书，以 HTTPS 启动（Demo/浏览器均要求 HTTPS）**

```bash
# 生成自签名证书（本地测试用）
mkdir -p /tmp/omni_ssl
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
    -keyout /tmp/omni_ssl/key.pem -out /tmp/omni_ssl/cert.pem \
    -subj "/CN=localhost"

# 启动（-m 传 LLM，其它子模型自动推断）
export MODEL=/path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-omni-server \
    -m "$MODEL" --host 0.0.0.0 --port 28099 -c 4096 -ngl 99 \
    --ssl-cert-file /tmp/omni_ssl/cert.pem \
    --ssl-key-file  /tmp/omni_ssl/key.pem
```

**方式 B：配置时关闭 OpenSSL，以普通 HTTP 启动**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF
cmake --build build --config Release -j --target llama-omni-server
# 之后启动不需要 --ssl-* 参数，用 http:// 访问
```

其它注意事项：
- **端口占用**：若端口已被其它进程占用，服务会绑定失败并退出。请更换端口或先释放占用。
- **模型懒加载**：服务本身启动很快，但模型在首个 WebSocket 会话 / 首次 `omni_init` 调用时才加载，因此首次调用会有 10–60s 的加载耗时。

验证启动成功：

```bash
# HTTPS（方式 A）用 -k 跳过自签证书校验
curl -sk https://127.0.0.1:28099/health
# → {"engine":"comni","status":"ok"}
```

### 5.2 用 WebSocket `/backend` 调用（现代协议 / Demo 所用）

消息格式：

- 上行首帧 `session.init`（`payload.mode` = `turn_based` 或 `full_duplex`，`payload.use_tts` 控制是否出声）
- 上行 `input.append`（`turn_based` 用 `input.messages`；`full_duplex` 用 `input.audio` + 可选 `input.video_frames`）
- 下行事件：`session.created` → `response.output.delta`（`kind` = `text`/`audio`/`listen`）→ `response.done` → `session.closed`

一个最小的 `turn_based` 纯文本示例（Python，`pip install websockets`）：

```python
import asyncio, json, ssl, websockets

async def main():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # 自签证书用；HTTP 模式改用 ws:// 且去掉 ssl 参数
    async with websockets.connect("wss://127.0.0.1:28099/backend", ssl=ctx, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "session.init",
            "payload": {"mode": "turn_based", "use_tts": False}
        }))
        print(json.loads(await ws.recv())["type"])   # session.created

        await ws.send(json.dumps({
            "type": "input.append",
            "input": {
                "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
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
                print("回复:", m.get("text")); break

asyncio.run(main())
```

- `turn_based` 可在 `messages[].content` 里带 `image`/`audio`/`video` 内容项做多模态问答；`input.use_tts_template=true` 时回复会带合成语音（`response.done.audio`，base64 float32 PCM）。
- `full_duplex` 每次 `input.append` 送 1 秒音频（+可选视频帧），服务端按 1Hz 决定说话/倾听，用于实时对话。

### 5.3 用 Legacy HTTP（SSE）调用（curl 手动调试）

调用顺序：`omni_init` → 循环（`prefill` → `decode`）。`decode` 返回 SSE 文本流，合成语音写到 `output_dir` 下。

```bash
export B=https://127.0.0.1:28099    # HTTP 模式改成 http:// 并去掉 -k

# 1) 初始化（首次会加载模型，内部已完成 index=0 的系统 prompt prefill）
curl -sk -X POST $B/v1/stream/omni_init -H 'Content-Type: application/json' \
    -d '{"media_type":1,"use_tts":true,"output_dir":"./tools/omni/output_server"}'

# 2) 发送一段用户音频（audio_path_prefix 为服务端可见的文件路径，cnt 为序号，从 1 开始）
curl -sk -X POST $B/v1/stream/prefill -H 'Content-Type: application/json' \
    -d '{"audio_path_prefix":"tools/omni/assets/test_case/audio_test_case/audio_test_case_0000.wav","cnt":1}'

# 3) 触发生成（SSE 流）
curl -sk -N -X POST $B/v1/stream/decode -H 'Content-Type: application/json' \
    -d '{"stream":true,"debug_dir":"./tools/omni/output_server"}'
```

SSE 返回形如（注意文本字段是 **`content`** 而非 `text`）：

```
data: {"content":"从前有座山，山里有座庙","end_of_turn":false,"is_listen":false,"stop":false}
data: {"content":"","end_of_turn":true,"is_listen":false,"stop":true}
data: [DONE]
```

| 字段 | 说明 |
| --- | --- |
| `content` | 文本片段（可能为空，展示前先过滤） |
| `is_listen` | `true` 表示模型转入倾听状态 |
| `stop` | `true` 表示本轮生成结束 |

> `media_type`：`1` = 纯音频，`2` = 音频+视觉；`prefill` 的 `cnt=0` 已由 `omni_init` 内部处理，自己的计数从 `1` 开始。

---

# 第二部分：Demo

MiniCPM-o 4.5 的 Demo 是一套**全双工音视频实时对话**的 Web 应用，分为在线体验与本地部署两种方式。

## 1. HuggingFace Space（在线体验）

打开即可在线体验官方托管的 Demo：<https://huggingface.co/spaces/openbmb/MiniCPM-o-4_5-Demo>（或直接访问 <https://minicpmo45.modelbest.cn/>）。

## 2. 本地部署：MiniCPM-o-Demo（配合 llama.cpp-omni C++ 后端）

如需在本机基于自建的 `llama-omni-server` 运行完整的音视频通话前端，使用官方 Demo 仓库：

- 仓库：<https://github.com/OpenBMB/MiniCPM-o-Demo>（使用默认的 `main` 分支）

> 分支说明：C++ 后端（llama.cpp-omni）已合并进 `main`，`main` 同时支持 PyTorch 后端和 C++ 后端。旧 `Comni` 分支已废弃，统一使用 `main`。

Demo 采用 gateway + worker + backend 架构，worker/gateway 是通用的 Python 编排层，后端可选 PyTorch（`py_backend/server.py`）或 C++（llama.cpp-omni 的 `llama-omni-server`）：

```
浏览器 ⇄ gateway.py（HTTPS，:8006）⇄ worker.py（:22400）⇄ llama-omni-server（HTTP，:22500）
```

### 方式一：Docker Compose（官方推荐）

Demo 为 C++ 后端提供了 `docker-compose.cpp.yml`，一条命令启动 gateway + cpp-worker-backend。cpp-worker-backend 镜像会在容器内自动克隆并编译 `tc-mb/llama.cpp-omni`，无需本机预先构建。

```bash
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git
cd MiniCPM-o-Demo

# 生成自签名证书（gateway 的 HTTPS 用）
mkdir -p certs data
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
    -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"

# 构建并启动（GGUF_MODEL_HOST_PATH 指向宿主机的 GGUF 目录，只读挂载）
GGUF_MODEL_HOST_PATH=/path/to/MiniCPM-o-4_5-gguf \
GATEWAY_HOST_PORT=8006 \
CPP_GPU_ID=0 \
docker compose -f docker-compose.cpp.yml up -d --build

docker compose -f docker-compose.cpp.yml logs -f cpp-worker-backend   # 看后端加载
# 浏览器打开 https://<host>:8006/
```

Docker 方式注意事项：
- **需要能访问 Docker Hub**：镜像基于 `docker/dockerfile:1` 与 `nvidia/cuda:*` 基础镜像；若机器访问不了 `docker.io`（离线/国内环境常见），构建会在拉取基础镜像时超时失败。需自行配置 registry 镜像源或改用方式二。
- **CUDA 架构**：`docker/Dockerfile.cpp-worker-backend` 默认 `CMAKE_CUDA_ARCHITECTURES=80;120`（A100 / Blackwell），未包含 Ada（sm_89，如 RTX 40 系列）。若运行时报 `no kernel image is available for execution on the device`，在该 Dockerfile 的 arg 中加上 `89` 重新构建。
- 证书只在 **gateway** 层使用（HTTPS 终结在 gateway）；容器内的 `llama-omni-server` 是普通 HTTP，无需证书（见下方说明）。

### 方式二：Bare-metal（无需 Docker）

适用于调试或无法访问 Docker Hub 的场景。三个进程依次启动：

```bash
# 0) 先构建 llama.cpp-omni 的 llama-omni-server（见第一部分第 1 节）
#    后端需为普通 HTTP 版：demo 的 worker 用 ws:// 连后端、且不会跳过自签证书校验，
#    连不上 HTTPS 后端。因此构建时需关闭 OpenSSL：
#       cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF
#       cmake --build build --config Release -j --target llama-omni-server

git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git
cd MiniCPM-o-Demo

# 1) 装 worker/gateway 依赖（C++ 后端模式无需 torch，轻量）
python3 -m venv .venv-cpp && . .venv-cpp/bin/activate
pip install "fastapi>=0.128.0" "httpx>=0.28.0" "numpy>=2.2.0" "pydantic>=2.11.0" \
            python-multipart "uvicorn>=0.40.0" "websockets>=16.0"

export MODEL=/path/to/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf

# 2) 启动后端 llama-omni-server（HTTP，:22500）
CUDA_VISIBLE_DEVICES=0 /path/to/llama.cpp-omni/build/bin/llama-omni-server \
    -m "$MODEL" --host 127.0.0.1 --port 22500 -c 8192 -ngl 99 &

# 3) 启动 worker（remote backend 模式，指向后端）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python worker.py \
    --host 0.0.0.0 --port 22400 --gpu-id 0 \
    --backend-server-url http://127.0.0.1:22500 &

# 4) 启动 gateway（本地调试可用 --http；浏览器摄像头/麦克风需改回默认 HTTPS）
python gateway.py --host 0.0.0.0 --port 8006 --http --workers localhost:22400 &

# 5) 验证
curl http://127.0.0.1:22500/health   # 后端: {"engine":"comni","status":"ok"}
curl http://127.0.0.1:22400/health   # worker: {"status":"healthy","worker_status":"idle",...}
curl http://127.0.0.1:8006/health    # gateway: {"status":"healthy",...}
# 浏览器打开 http://<host>:8006/
```

注意事项：
- 后端需为普通 HTTP 版（`-DLLAMA_OPENSSL=OFF`）。这也解释了官方 Docker 镜像为何不安装 `libssl-dev`——正是为了让 `llama-omni-server` 编为 HTTP 版。
- **端口冲突**：worker、后端、gateway 默认端口（22400 / 22500 / 8006）在共享机器上可能冲突；被占用时进程会静默退出，请改用空闲端口。
- **摄像头/麦克风需 HTTPS**：gateway 用 `--http` 时浏览器会禁用媒体设备、只能文本输入；正式使用去掉 `--http`（默认 HTTPS，需 `certs/cert.pem`、`certs/key.pem`）。
- worker 在 C++ 后端模式下不加载 PyTorch 模型，只把 `session.init / input.append` 透传给后端；对外 WS 路由为 `/v1/worker/chat`（turn_based）与 `/v1/worker/duplex`（full_duplex）。

更多细节参见 Demo 仓库的 `README.md` / `README_zh.md`（`main` 分支）及 `docker-compose.cpp.yml`、`docker/entrypoint-cpp-worker-backend.sh`。

---

## 参考链接

- llama.cpp-omni 仓库：<https://github.com/tc-mb/llama.cpp-omni>
- 模型（PyTorch）：<https://huggingface.co/openbmb/MiniCPM-o-4_5>
- 模型（GGUF）：<https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf> · <https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf>
- 在线 Demo：<https://minicpmo45.modelbest.cn/>
- Demo 仓库：<https://github.com/OpenBMB/MiniCPM-o-Demo>（`main` 分支）
- HF Space：<https://huggingface.co/spaces/openbmb/MiniCPM-o-4_5-Demo>
