# ocrcpm

一个可通过 pip 安装、面向 MiniCPM-V OCR 的**文档解析流水线**。

包本身主要负责流水线编排和 CLI。参考 MinerU 的**分层设计**，推理能力分为两层，而不是将所有方式平铺成一个“后端”列表：

1. **SDK 部署模式**——SDK 如何访问模型，类似 MinerU 的 `*-engine` 与 `*-http-client`：
   - `transformers-local`：通过 Hugging Face Transformers 在当前进程内加载 OCR 模型。
   - `http-client`：调用一个已经运行的 OpenAI-compatible 接口。
   - `vllm`（自动启动）：SDK 启动本地 vLLM OpenAI 服务，再调用该服务。
   - `sglang`（自动启动）：SDK 启动本地 SGLang OpenAI 服务，再调用该服务。
   - `llama-cpp`（自动启动）：SDK 启动本地 `llama-server`，再调用该服务。
   - `ollama`（自动启动）：SDK 启动本地 `ollama serve`，必要时用 GGUF 注册模型，再通过原生 `/api/chat` 调用。
2. **推理引擎**——真正执行模型推理的实现，类似 MinerU 的 pip extras / 内部引擎。可以是进程内 Transformers、vLLM、SGLang、项目内置的 HF Transformers 服务、llama.cpp、Ollama，或者任意远程 OpenAI-compatible 服务。重量级引擎通过可选依赖安装。

```text
ocrcpm
  -> CLI / Python API
  -> 文档流水线（layout -> prepare -> infer）
  -> 部署模式
       -> transformers-local（在当前 Python 进程加载模型）
       -> http-client       （调用 OpenAI-compatible 接口）
       -> vllm              （自动启动本地 vLLM OpenAI 服务）
       -> sglang            （自动启动本地 SGLang OpenAI 服务）
       -> llama-cpp         （自动启动本地 llama-server）
       -> ollama            （自动启动本地 ollama serve）
  -> 推理实现
       -> 进程内 Transformers / vLLM / SGLang / Transformers 服务 / llama.cpp / Ollama / 远程服务
```

> `transformers-local` 是最简单的单进程冒烟测试后端。
> `llama.cpp` 既可由 SDK 自动启动，也可作为外部 OpenAI-compatible 服务运行。
> `ollama` 复用与 llama.cpp 相同的 GGUF 文件，但不需要自行编译。

## 流水线

```text
输入（PDF / PNG / JPG）
  -> parse-layout：PP-DocLayoutV3 检测版面块（bbox + label）并保存裁块
  -> prepare     ：收集版面结果，生成 page_tasks.jsonl
  -> infer       ：通过本地后端或 OpenAI-compatible 接口逐块执行 OCR
  -> output      ：生成 *.md、results_json/*.json 以及耗时/指标文件
```

## 安装

核心安装刻意保持轻量，不包含 PyTorch。只在需要本地执行模型时安装相应依赖：

```bash
pip install ocrcpm
```

按需安装推理引擎和功能，方式与 MinerU 的可选依赖类似：

```bash
pip install "ocrcpm[layout]"    # PP-DocLayoutV3 版面解析（torch + transformers）
pip install "ocrcpm[pdf]"       # PDF 渲染（pymupdf）
pip install "ocrcpm[server]"    # 内置 HF Transformers OpenAI 服务
pip install "ocrcpm[vllm]"      # 本地 vLLM 服务后端
pip install "ocrcpm[sglang]"    # 本地 SGLang 服务后端
pip install "ocrcpm[llamacpp]"  # llama.cpp HTTP 客户端（llama-server 二进制需单独提供）
pip install "ocrcpm[ollama]"    # Ollama HTTP 客户端（ollama 二进制需单独安装）
pip install "ocrcpm[models]"    # Hugging Face / ModelScope 模型下载工具
pip install "ocrcpm[all]"       # layout + pdf + server（不包含 vllm/sglang/llamacpp/ollama）
```

> 注意：版面解析依赖 `torch` 和 `transformers`。即使 OCR 模型部署在远程服务，完整本地流水线也至少需要安装 `layout` 可选依赖。

## 模型（不随安装包分发）

模型权重**不会**打包进 pip 安装包，需要单独下载，并在配置中填写本地路径：

- 版面模型：`PP-DocLayoutV3_safetensors`
- OCR 模型：HF 格式的 MiniCPM-V OCR 模型目录

可通过以下命令下载模型快照并生成可运行配置：

```bash
ocrcpm download-models \
  --ocr-repo <your-org-or-user/minicpmv-ocr-repo> \
  --output-dir ~/.cache/ocrcpm/models \
  --config-output my_config.yaml \
  --backend hf-server
```

版面模型默认使用 `PaddlePaddle/PP-DocLayoutV3_safetensors`。
OCR 模型仓库需要通过 `--ocr-repo` 显式指定，因为很多 OCR checkpoint 是私有的，或者只用于特定部署。国内网络可使用 ModelScope，并填写对应仓库 ID：

```bash
ocrcpm download-models \
  --source modelscope \
  --layout-repo <modelscope-layout-repo> \
  --ocr-repo <modelscope-ocr-repo> \
  --config-output my_config.yaml
```

如果模型已经存在于本地，可以只生成配置：

```bash
ocrcpm download-models \
  --skip-download \
  --layout-dir /abs/path/to/PP-DocLayoutV3_safetensors \
  --ocr-dir /abs/path/to/ocr_minicpmv_model \
  --config-output my_config.yaml
```

## 快速开始：本地 Transformers 后端

完整的本地安装、后端切换与排错步骤见：
[`../minicpm-v-ocr_pip_zh.md`](../minicpm-v-ocr_pip_zh.md)。

生成默认配置，在当前 Python 进程中直接加载 OCR 模型：

```bash
ocrcpm init \
  --skip-download \
  --layout-dir /abs/path/to/PP-DocLayoutV3_safetensors \
  --ocr-dir /abs/path/to/ocr_minicpmv_model \
  --backend transformers-local
```

运行完整流水线：

```bash
ocrcpm parse --input /abs/path/doc.pdf --output out.md
```

## 快速开始：Ollama 后端

Ollama 内置的 llama.cpp 已包含 MiniCPM-V4.x 适配，可直接复用 llama.cpp 后端的那组 GGUF 文件，无需自行编译：

```bash
ocrcpm init \
  --skip-download \
  --layout-dir /abs/path/to/PP-DocLayoutV3_safetensors \
  --backend ollama \
  --ollama-model-gguf /abs/path/to/ocr_minicpmv_gguf/model-f16.gguf \
  --ollama-mmproj /abs/path/to/ocr_minicpmv_gguf/mmproj-f16.gguf \
  --model-name minicpm-v4.7 \
  --gpu-id 0 \
  --port 11434

ocrcpm parse --input /abs/path/doc.pdf --output out.md
```

首次运行时 OCRCPM 会启动 `ollama serve`，并用两个 GGUF 自动执行一次 `ollama create`。如果模型已经注册过，省略两个 GGUF 参数即可。`ollama` 二进制可通过 `--ollama-bin` 或 `OCRCPM_OLLAMA_BIN` 指定。

`ollama` 使用官方预编译二进制，不需要源码或自行编译，要求 0.32.6-rc0 或更高版本，从 [Releases](https://github.com/ollama/ollama/releases) 下载解压即可（解压后 `bin` 与 `lib` 的相对位置不能改动）。

## 快速开始：SGLang 后端

使用独立的 SGLang 环境自动启动服务。注意官方 SGLang 目前只支持到 MiniCPM-V4.6，运行 4.7 需要带有 4.7 适配的 SGLang 源码（可用 [tc-mb/sglang 的 `support-minicpm-ocr` 分支](https://github.com/tc-mb/sglang/tree/support-minicpm-ocr)），并以 `pip install -e` 装入该环境：

```bash
ocrcpm init \
  --skip-download \
  --layout-dir /abs/path/to/PP-DocLayoutV3_safetensors \
  --ocr-dir /abs/path/to/ocr_minicpmv_model \
  --backend sglang \
  --sglang-python /abs/path/to/sglang-env/bin/python \
  --cuda-home /abs/path/to/cuda-toolkit \
  --model-name minicpm-v4.7 \
  --gpu-id 0

ocrcpm parse --input /abs/path/doc.pdf --output out.md
```

也可以设置 `OCRCPM_SGLANG_PYTHON`，省略命令中的
`--sglang-python`。服务日志写入本次运行目录的 `logs/engine.log`。

## 快速开始：http-client 后端

1. 启动承载 OCR 模型的 OpenAI-compatible 服务。如果 vLLM 暂不支持当前模型，可以使用项目内置的 HF 服务：

```bash
pip install "ocrcpm[server]"
ocrcpm-server \
  --model-path /abs/path/to/ocr_minicpmv_model \
  --host 127.0.0.1 --port 18599
```

2. 复制示例配置并填写路径：

```bash
python -c "import importlib.resources as r, shutil; \
shutil.copy(r.files('ocrcpm')/'configs'/'example_http_client.yaml', 'my_config.yaml')"
```

3. 运行完整流水线：

```bash
ocrcpm run --config my_config.yaml
# 或者单文件一键解析为 Markdown：
ocrcpm parse --config my_config.yaml --input /abs/path/doc.pdf --output out.md
```

## 完整流水线 OpenAI 服务

`ocrcpm-server` 暴露的是裁块级 OCR 模型。不要让文档前端直接将整页图片发送到该接口，否则会绕过版面检测和 Markdown 组装。

如果需要将 `ocrcpm parse` 使用的完整流水线暴露为 OpenAI-compatible 接口，请先按前文启动裁块 OCR 后端，并在 `my_config.yaml` 中通过 `http-client` 模式指向它，然后运行：

```bash
ocrcpm-pipeline-server \
  --config my_config.yaml \
  --host 0.0.0.0 \
  --port 18600
```

该服务在 `http://127.0.0.1:18600/v1` 暴露模型 `ocrcpm-pipeline`。其 `/v1/chat/completions` 接口接收一张图片的 data URL，执行版面检测 → 裁块 OCR → Markdown 组装，并同时支持普通 JSON 与 OpenAI-compatible SSE 流式响应（`stream: true`）。

Open WebUI 或其他文档前端应连接该接口，而不是裁块级 OCR 接口。

## CLI

```bash
ocrcpm show-config   --config <yaml>   # 输出合并后的配置
ocrcpm parse-layout  --config <yaml>   # 阶段 1：仅执行版面解析
ocrcpm prepare       --config <yaml>   # 阶段 2：生成页面任务
ocrcpm infer         --config <yaml>   # 阶段 3：OCR + Markdown
ocrcpm run           --config <yaml>   # 完整流水线（1 -> 2 -> 3）
ocrcpm init          --layout-dir DIR [--ocr-dir DIR] [--backend transformers-local|vllm-auto|sglang|llama-cpp|ollama]
ocrcpm parse         --input FILE [--output OUT.md]
ocrcpm download-models --ocr-repo <repo> --config-output my_config.yaml
```

控制台命令为 `ocrcpm`，也可以通过 `python -m ocrcpm` 运行。

## 部署模式与推理引擎配置

SDK 提供四种部署模式（第一层）。推理引擎（第二层）可以是在当前进程运行的 Transformers，也可以是 OpenAI-compatible 接口背后的任意推理服务。

- **transformers-local：** 设置 `engine.type: minicpm_transformers_local`。SDK 在当前进程加载 MiniCPM-V OCR 模型，并直接调用 `model.chat(...)`。这是最简单的本地冒烟测试方式。
- **http-client：** 设置 `engine.type: minicpm_vllm_openai_api`，并在 `engine.server_urls` 中填写一个或多个 OpenAI-compatible 服务地址。请求会在多个地址间随机分配并自动故障转移。SDK 不会在该模式下启动本地服务。接口背后的引擎可以是 vLLM、内置 HF Transformers 服务、llama.cpp 或远程服务；SDK 只使用 OpenAI HTTP API。
- **vllm（自动启动）：** 设置 `engine.type: minicpm_vllm_openai_api`，保持 `server_urls` 为空，并设置 `auto_start: true` 和 `engine.vllm_python`，也可使用环境变量 `OCRCPM_VLLM_PYTHON`。该环境中的 vLLM 必须带有本模型的适配，官方版本只支持到 4.6；MiniCPM-V4.7 可使用 [tc-mb/vllm 的 `tc-ocr` 分支](https://github.com/tc-mb/vllm/tree/tc-ocr)。适配版本按 checkpoint `config.json` 里的 `version` 字段分派，使用前需确认该字段与所用 vLLM 的预期一致。
- **sglang（自动启动）：** 设置 `engine.type: minicpm_sglang_openai_api`，保持 `server_urls` 为空，并设置 `auto_start: true` 和 `engine.sglang_python`，也可使用环境变量 `OCRCPM_SGLANG_PYTHON`。OCRCPM 会传入 V4.7 的 `4x` 配置并隔离 TVM-FFI JIT 缓存。该环境中的 SGLang 必须带有 MiniCPM-V4.7 适配，官方版本只支持到 4.6；4.7 可使用 [tc-mb/sglang 的 `support-minicpm-ocr` 分支](https://github.com/tc-mb/sglang/tree/support-minicpm-ocr)。
- **llama.cpp（自动启动）：** 设置 `engine.type: minicpm_llamacpp_openai_api`、`engine.llama_server`、语言模型 GGUF 和 `mmproj_path`；也可通过 `OCRCPM_LLAMA_SERVER` 指定二进制。设置 `auto_start: false` 和 `server_urls` 时可连接外部 llama.cpp 服务。原版上游 llama.cpp 即可加载已转换好的 GGUF，不需要补丁，建议使用 b10238 或更高版本；上游 Linux 发布包不含 CUDA 版本，GPU 环境需自行用 `-DGGML_CUDA=ON` 编译。
- **Ollama（自动启动）：** 设置 `engine.type: minicpm_ollama_api`，并把 `engine.model_name` 设为 Ollama 中的模型名。同时给出 `engine.model_path` 和 `engine.mmproj_path` 时，OCRCPM 会在模型缺失的情况下自动执行 `ollama create`；两者都留空则要求模型已经注册。`engine.ollama_bin`（或 `OCRCPM_OLLAMA_BIN`）指定二进制，`engine.ollama_models_dir` 覆盖 Ollama 的存储目录。设置 `auto_start: false` 和 `server_urls` 时可连接外部 Ollama 服务，此时任务结束会主动卸载模型释放显存。该后端使用 Ollama 原生 `/api/chat` 接口，因为其 OpenAI 兼容接口不支持 OCR 所需的 `top_k` 和 `repeat_penalty`。

## Python API

```python
from ocrcpm import load_config, parse_layout_run, prepare_run, infer_run

cfg = load_config("my_config.yaml")
parse_layout_run(cfg)
prepare_run(cfg)
summary = infer_run(cfg)
print(summary)
```

## 输出目录

```text
<run_root>/<run_tag>/
  output/*.md               # 最终 Markdown
  results_json/*.json       # 各版面块的 bbox / label / 文本 / 状态
  summary.json
  per_page_metrics.{jsonl,csv}
  runtime/
    page_tasks.jsonl
    {layout,page,crop}_timing.jsonl
    intermediate/layouts/*.json
    intermediate/crops/<image_id>/*.png
```
