# MiniCPM-V OCR - pip (ocrcpm)

> [!NOTE]
> `ocrcpm` 封装的是一条**流水线**，不是单次调用的包装：先在 CPU 上做版面检测，把每个版面块裁出来，逐块送给 OCR 后端识别，最后组装成 Markdown 和 JSON。版面检测固定在 CPU 上运行，可替换的只有 OCR 模型的推理后端。
>
> 本目录已包含可直接安装的 wheel（`ocrcpm-0.1.0-py3-none-any.whl`）和用于构建它的源码（`wheel_source/`）。

> [!IMPORTANT]
> OCR 模型尚未开源，获取方式待开源后在此补充。下文命令中的模型目录一律用占位路径 `$PWD/layout_model`、`$PWD/ocr_model`、`$PWD/ocr_gguf` 表示，请替换为实际路径。模型权重不会打进 wheel。

## 0. 后端选择与运行环境

OCR 模型共支持五种推理后端，安装成本和性能差别很大：

| 后端 | 安装成本 | 说明 |
| --- | --- | --- |
| transformers-local | 无，装完 wheel 即可用 | 最简单，速度最慢，适合先跑通流程（第 4、5 节） |
| Ollama | 下载官方预编译二进制 | 不用编译，用 GGUF，性价比最高（第 7 节） |
| llama.cpp | 需自行编译（约十几分钟） | 同样用 GGUF，比 Ollama 多一步编译（第 6 节） |
| SGLang | 纯 Python 包，`pip install -e` | 无需编译 CUDA 扩展，吞吐高（第 8 节） |
| vLLM | 需编译 CUDA 扩展（约 1 小时） | 吞吐高，安装最费事（第 9 节） |

建议先按第 2～5 节用 transformers-local 跑通，确认模型和输入文件都没问题，再按需要切换到其他后端。

五个后端在同一页论文首页（切成 11 块）上比对过识别结果：11 个版面块中有 10 块完全一致，摘要块在各后端之间存在差异。

运行环境要求：

- Python 3.10～3.12（已在 3.12 上验证）；
- NVIDIA 显卡，显存 24 GB 及以上。24 GB 是实测能跑通的下限，其中 vLLM 后端需要按第 9 节调小 `--max-num-batched-tokens`；
- 使用 llama.cpp、SGLang、vLLM 后端时需要 CUDA 工具链（含 `bin/nvcc`），Ollama 和 transformers-local 不需要；
- 磁盘：模型和框架源码合计约 40 GB，其中 vLLM 编译时额外拉取约 3 GB 第三方依赖，Ollama 的模型存储另需约 8 GB。

## 1. 准备工作目录

新建一个空目录，把 wheel、模型和测试文件放进去：

```bash
mkdir -p ~/ocrcpm_demo
cd ~/ocrcpm_demo
```

最终结构应为：

```text
ocrcpm_demo/
├── ocrcpm-0.1.0-py3-none-any.whl
├── layout_model/                  # 版面检测模型（PP-DocLayoutV3，safetensors 格式）
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── ocr_model/                     # OCR 模型，HuggingFace 格式
│   ├── config.json
│   ├── modeling_minicpmv.py
│   ├── *.safetensors
│   └── ...
├── ocr_gguf/                      # llama.cpp / Ollama 后端可选
│   ├── model-f16.gguf
│   └── mmproj-f16.gguf
└── demo.png
```

两个模型目录必须完整复制，不能只复制其中的权重文件——配置文件和随模型分发的胶水代码同样是运行所必需的。llama.cpp 和 Ollama 两个后端复用同一组 GGUF 文件。

推理框架本身都不在这里，需要联网获取：llama.cpp 从上游获取源码后自行编译（第 6 节），Ollama 直接下载官方预编译二进制（第 7 节），SGLang 和 vLLM 需要从带 OCR 适配的分支获取（第 8、9 节）。

## 2. 创建 Python 环境

推荐 Python 3.10～3.12，并使用独立环境：

```bash
cd ~/ocrcpm_demo
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

部分 Debian/Ubuntu 的系统 Python 没有自带 venv 模块，`python -m venv` 会提示 `ensurepip is not available`，按提示安装即可（例如 `apt install python3.12-venv`），或者改用 Conda：

```bash
conda create -n ocrcpm-demo python=3.12 -y
conda activate ocrcpm-demo
cd ~/ocrcpm_demo
```

## 3. 安装 wheel

```bash
python -m pip install "./ocrcpm-0.1.0-py3-none-any.whl[all]"
```

确认命令行入口可用：

```bash
ocrcpm --help
```

`[all]` 会安装版面解析、Transformers 推理及 PDF 解析依赖，但不会安装 vLLM、SGLang 和 llama.cpp。核心依赖刻意保持轻量，只有 pyyaml、requests 和 pillow，其余按可选依赖划分：

| extra | 装什么 | 用途 |
| --- | --- | --- |
| `layout` | torch、torchvision、transformers、opencv-headless | 版面解析 |
| `pdf` | pymupdf | PDF 输入渲染 |
| `server` | torch、transformers、accelerate、einops | 内置的 HF Transformers OpenAI 兼容服务 |
| `vllm` | vllm | 本地 vLLM 服务后端 |
| `sglang` | sglang[all] | 本地 SGLang 服务后端 |
| `llamacpp` / `ollama` | 空 | 这两个后端依赖外部二进制，不是 Python 包 |
| `models` | huggingface_hub、modelscope | 模型下载工具 |
| `all` | layout + pdf + server | 不含 vllm / sglang / llamacpp / ollama |

> 上述安装过程需要能够访问 pip 软件源。完全离线部署时，除了 wheel 和模型，还需要提前提供依赖 wheel，或者直接提供已经装好的 Conda 环境。

### 3.1 从源码构建 wheel（可选）

本目录的 `wheel_source/` 就是构建这个 wheel 的全部源码，采用 src 布局、setuptools 构建。需要改代码后重新打包时：

```bash
cd deployment/pip/wheel_source
python -m pip install build
python -m build
```

产物在 `dist/` 下，包含一个 `py3-none-any` 的 wheel 和一个 sdist。包内是纯 Python，没有编译扩展，wheel 只有几十 KB——模型权重和推理框架都不在里面。也可以直接以可编辑方式安装，改完立即生效：

```bash
python -m pip install -e ".[all]"
```

## 4. 生成默认配置

以下是最简单的本地 Transformers 运行方式，不需要先启动模型服务。在 `~/ocrcpm_demo` 目录执行：

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

该命令不会下载模型，只检查本地模型目录并生成默认配置：

```text
~/.config/ocrcpm/config.yaml
```

初始化成功后，后续执行 `ocrcpm parse` 不需要再传 `--config`。

这里的 `--input` 只是写进配置里的占位路径，`ocrcpm init` 本身不执行识别，真正处理哪个文件由第 5 节 `ocrcpm parse --input` 决定，两者可以不同。更换 GPU 时把 `--gpu-id 0` 改为对应编号。完整参数列表见 `ocrcpm init --help`。

## 5. 执行 OCR

解析图片：

```bash
ocrcpm parse --input "$PWD/demo.png" --output "$PWD/demo.md"
```

解析 PDF：

```bash
ocrcpm parse --input "$PWD/demo.pdf" --output "$PWD/demo.md"
```

命令输出中应能看到类似结果：

```text
"counts": {
  "planned": 1,
  "ok": 1,
  "failed": 0
}
```

最终 Markdown 位于 `~/ocrcpm_demo/demo.md`，流水线中间结果默认位于 `~/ocrcpm_demo/runs/`：

```text
<run_root>/<run_tag>/
  output/*.md               # 最终 Markdown
  results_json/*.json       # 每个版面块的 bbox / label / 文本 / 状态
  summary.json
  per_page_metrics.{jsonl,csv}
  runtime/
    page_tasks.jsonl
    {layout,page,crop}_timing.jsonl
    intermediate/layouts/*.json
    intermediate/crops/<image_id>/*.png
```

流水线固定为三段，中间产物落盘，因此任意一段都可以单独重跑：

```bash
ocrcpm show-config  --config <yaml>   # 输出合并后的配置
ocrcpm parse-layout --config <yaml>   # 阶段 1：版面检测，保存 bbox、label 和裁块
ocrcpm prepare      --config <yaml>   # 阶段 2：汇总为 page_tasks.jsonl
ocrcpm infer        --config <yaml>   # 阶段 3：逐块 OCR，组装 Markdown
ocrcpm run          --config <yaml>   # 完整流水线（1 -> 2 -> 3）
```

`ocrcpm` 也可以通过 `python -m ocrcpm` 调用。

## 6. 使用 llama.cpp 后端（可选）

> 第 6～9 节各有一条 `ocrcpm init`，它们写的都是同一个 `~/.config/ocrcpm/config.yaml`，后执行的会覆盖先执行的。也就是说同一时刻只有一个后端生效，要换回其他后端，重新执行对应那节的 `ocrcpm init` 即可。如果想让多个后端配置并存，给每条 `ocrcpm init` 加上 `--config-output "$PWD/ocrcpm_<后端名>.yaml"`，用的时候通过 `ocrcpm parse --config` 指定。

llama.cpp 不通过 pip 安装，请自行从上游获取源码。运行本模型所需的 `qwen35` 架构和 `minicpmv4_6` 投影仪，上游最早在 b10091 就已具备；本项目基于 b10238 验证，建议使用该版本或更高版本。转换好的 GGUF 用原版上游代码即可直接加载，不需要任何补丁。

上游只为 Windows 提供 CUDA 预编译包，Linux 的发布包仅含 CPU/Vulkan 版本，因此在 Linux + NVIDIA 环境下需要自行编译：

```bash
git clone --depth 1 --branch b10238 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
cd ..
```

如果不希望编译，改用第 7 节的 Ollama 后端，它提供现成的 Linux CUDA 二进制。

确认服务程序已经生成，然后生成配置：

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

之后仍使用同一条命令执行 OCR：

```bash
ocrcpm parse --input "$PWD/demo.png" --output "$PWD/demo.md"
```

OCRCPM 会自动启动 `llama-server`、等待服务就绪，并在任务结束后关闭服务。服务日志位于本次运行目录的 `logs/engine.log`。生成的配置会自动设置 `repeat_penalty=1.1`，避免长文本重复。

上面的命令没有 `--ocr-dir`，这是有意的：llama.cpp 和 Ollama 直接加载 GGUF，用不到 HF 格式的模型目录。这里 `--max-model-len` 用 8192 而不是后面 SGLang / vLLM 的 40960，是因为 GGUF 后端按整个上下文长度一次性分配 KV cache，取值过大会显著增加显存占用；而 OCRCPM 是按版面切块逐块识别的，单块内容用 8192 足够。

版面检测仍由 OCRCPM 在 CPU 上运行，只有 OCR 模型使用 llama.cpp。

## 7. 使用 Ollama 后端（可选）

Ollama 已经把内置的 llama.cpp 升级到包含 MiniCPM-V4.x 适配的版本，因此可以直接复用第 6 节的那组 GGUF 文件，不需要自己编译 llama.cpp。

Ollama 官方发布的就是预编译二进制，不需要源码，也不需要在目标机器上编译。版本要求为 0.32.6-rc0 或更高，本项目在该版本上做过完整验证。先确认目标机器上的 `ollama` 可用：

```bash
ollama --version
```

服务没启动时这条命令会打印 `ollama version is 0.0.0` 并附带一行 `Warning: client version is 0.32.6-rc0`，属于正常现象，后面那个才是真实版本号。

如果没有安装，从官方 Release 下载压缩包（x86_64 + NVIDIA 显卡对应 `ollama-linux-amd64.tar.zst`，约 1.4 GB）并解压：

```bash
curl -fL -o ollama-linux-amd64.tar.zst \
  https://github.com/ollama/ollama/releases/download/v0.32.6-rc0/ollama-linux-amd64.tar.zst
mkdir -p ollama-root
tar --use-compress-program=unzstd -xf ollama-linux-amd64.tar.zst -C ollama-root
```

官方只提供 zstd 压缩包，如果机器上没有 `unzstd`，先装 zstd（`apt install zstd` 或 `yum install zstd`）。ARM64、ROCm 等其他架构的包和更高版本见 <https://github.com/ollama/ollama/releases>。解压后的目录结构如下，约 2.1 GB：

```text
ollama-root/
├── bin/ollama                 # 主程序
└── lib/ollama/                # ggml 运行库，含 cuda_v12 / cuda_v13 两套 CUDA 后端
```

`bin` 和 `lib` 的相对位置不能改动，主程序按相对路径查找 `../lib/ollama`。不必装进系统目录，在初始化命令中用 `--ollama-bin /abs/path/to/ollama-root/bin/ollama` 指定绝对路径即可。如果磁盘紧张，可以只保留与显卡驱动匹配的那一套 CUDA 后端目录（驱动 13.x 用 `cuda_v13`，12.x 用 `cuda_v12`），能省下约 1 GB。

生成 Ollama 后端配置：

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

之后仍使用同一条命令执行 OCR：

```bash
ocrcpm parse --input "$PWD/demo.png" --output "$PWD/demo.md"
```

首次运行时，OCRCPM 会启动 `ollama serve`，用两个 GGUF 文件自动执行一次 `ollama create`，等待模型就绪后再开始识别，任务结束时关闭服务。注册只在第一次发生，之后的运行会直接复用已注册的模型。服务日志和 `ollama create` 的输出都写入本次运行目录的 `logs/engine.log`。

注意 `ollama create` 会把 GGUF 复制进 Ollama 自己的存储目录（默认 `~/.ollama/models`）。注册过程中语言模型会先按原样存一份、再存一份改写过元数据的副本，且旧副本不会自动清理，实测 2.8 GB 的 GGUF 会占用约 4.7 GB。加上原始文件，请预留约 8 GB 空间。如果该分区空间不足，可以在初始化命令中追加 `--ollama-models-dir /abs/path/to/ollama_models`。

如果模型已经在 Ollama 中注册过，可以省略两个 GGUF 参数，OCRCPM 会直接使用 `--model-name` 对应的模型。手动注册时，Modelfile 中两行 `FROM` 分别指向语言模型和投影仪，Ollama 会自动识别哪一个是投影仪：

```text
FROM /abs/path/to/ocr_gguf/model-f16.gguf
FROM /abs/path/to/ocr_gguf/mmproj-f16.gguf
```

```bash
ollama create minicpm-v-ocr -f Modelfile
```

OCRCPM 通过 Ollama 的原生 `/api/chat` 接口请求，而不是 `/v1/chat/completions`。Ollama 的 OpenAI 兼容接口不支持 `top_k` 和 `repeat_penalty`，而 OCR 依赖这两个参数做贪心解码和抑制重复。

上面的命令同样不需要 `--ocr-dir`，理由和第 6 节一致。`--port 11434` 是 Ollama 的默认端口，如果机器上已经有别的 Ollama 服务占用，换一个端口即可。

## 8. 使用 SGLang 后端（可选）

不能用 `pip install sglang` 安装的官方版本。官方仓库和 PyPI 目前缺少本 OCR 模型的注册和配置解析，直接用官方版会在加载模型时失败。请使用带有适配的分支 [tc-mb/sglang 的 `support-minicpm-ocr` 分支](https://github.com/tc-mb/sglang/tree/support-minicpm-ocr)，它在官方代码基础上补齐了适配，改动只涉及模型类、配置和多模态处理器的注册，没有改动推理内核。

```bash
git clone --branch support-minicpm-ocr https://github.com/tc-mb/sglang.git "$PWD/sglang"
git -C "$PWD/sglang" checkout 775f12f0d
```

`775f12f0d` 是本项目验证过的提交。接着创建独立环境并以可编辑方式安装（要求 Python 3.10 及以上，已在 3.12 上验证）：

```bash
python -m venv "$PWD/sglang_env"
source "$PWD/sglang_env/bin/activate"
python -m pip install -U pip
python -m pip install -e "$PWD/sglang/python"
deactivate
```

SGLang 是纯 Python 包，本身不需要编译，但会拉取 torch、flashinfer 等依赖，视网络情况可能要十几分钟。装好后确认装的确实是这份源码，输出中应出现 `Editable project location` 且指向克隆下来的 `sglang/python`：

```bash
"$PWD/sglang_env/bin/python" -m pip show sglang | grep -E "Version|Editable"
```

由于是可编辑安装，`sglang/` 目录在部署后不能删除或移动，否则环境会失效。

生成 SGLang 后端配置：

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

`--cuda-home` 请指向一个带 `bin/nvcc` 的 CUDA 工具链目录（如果 nvcc 装在 conda 环境里，就填那个环境的路径）。配置了它之后，OCRCPM 会把对应的 CUDA runtime 加入动态库搜索路径，并按工具链隔离 TVM-FFI JIT 缓存，避免误用其他 CUDA 版本生成的缓存。

之后执行：

```bash
ocrcpm parse --input "$PWD/demo.png" --output "$PWD/demo.md"
```

OCRCPM 会自动启动 SGLang、等待 `/v1/models` 就绪，并在任务结束后关闭服务。也可通过 `OCRCPM_SGLANG_PYTHON` 指定环境 Python。服务日志写入本次运行目录的 `logs/engine.log`，可以据此确认走的是 OCR 模型的实现而不是回退到了别的版本。

24 GB 显存的卡上，`--max-model-len 40960` 配合默认的 `gpu_memory_utilization` 可以正常启动，无需额外调整。

## 9. 使用 vLLM 后端（可选）

官方 vLLM 同样无法加载本模型，请使用带有适配的分支 [tc-mb/vllm 的 `tc-ocr` 分支](https://github.com/tc-mb/vllm/tree/tc-ocr)。该分支按模型目录 `config.json` 里的 `version` 字段分派模型实现。

```bash
git clone --branch tc-ocr https://github.com/tc-mb/vllm.git "$PWD/vllm"
git -C "$PWD/vllm" checkout 87fcfae14
```

`87fcfae14` 是本项目验证过的提交。该分支除了模型支持，还带了一套挂在 `/v1/chat/completions` 上的服务端 OCR pipeline，但那部分只有在启动时传 `--ocr-layout-model` 才会启用；OCRCPM 自己在客户端做版面切分，不传该参数，服务端就是纯推理服务。

接着为 vLLM 创建独立环境并以可编辑方式安装。这一步需要编译 CUDA 扩展，请先准备好 CUDA 工具链（该提交要求 torch 2.13，对应 CUDA 13.0）：

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

`TORCH_CUDA_ARCH_LIST` 请填目标显卡的算力（RTX 4090 为 `8.9`，A100 为 `8.0`，可用 `nvidia-smi --query-gpu=compute_cap --format=csv` 查询）。不指定会为所有架构各编译一遍，耗时成倍增加。`MAX_JOBS` 按机器核数调整。

安装耗时主要在两处：cmake 会从 GitHub 拉取 cutlass、triton、flash-attention 等第三方依赖共约 3 GB，然后编译 CUDA 内核。在 96 核机器上限定单一架构时，编译约 25 分钟，加上依赖下载视网络情况从十几分钟到一小时不等。

`flashinfer` 会作为依赖自动装上。如果启动时报 `ModuleNotFoundError: No module named 'flashinfer'`，手动补装即可：

```bash
"$PWD/vllm_env/bin/python" -m pip install flashinfer-python nvidia-ml-py
```

与 SGLang 一样是可编辑安装，克隆出来的 `vllm/` 目录在部署后不能删除或移动。

生成 vLLM 后端配置：

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

`--cuda-home` 需要指向一个带 `bin/nvcc` 的 CUDA 工具链：flashinfer 的部分算子是启动时 JIT 编译的，找不到 nvcc 会报 `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`。

之后执行：

```bash
ocrcpm parse --input "$PWD/demo.png" --output "$PWD/demo.md"
```

OCRCPM 会以 `python -m vllm.entrypoints.openai.api_server` 启动服务，自动带上 `--trust-remote-code`、`--enforce-eager` 以及由配置推导出的 `--max-model-len`、`--limit-mm-per-prompt` 和 `--gpu-memory-utilization`，等待 `/v1/models` 就绪后开始识别，任务结束时关闭服务。也可通过 `OCRCPM_VLLM_PYTHON` 指定环境 Python。

上面命令里的 `--max-num-batched-tokens 8192` 和 `--gpu-memory-utilization 0.88` 是 24 GB 显卡上验证过的组合，请不要照搬 SGLang 那节的 40960：vLLM 启动时会按这个值预分配显存做一次 profiling，取 40960 会在这一步 OOM，而 `--max-model-len` 保持 40960 不受影响。显存更大的卡可以把它调高以提升吞吐。

如果 vLLM 服务已经在别处运行，可以不使用本节的自动启动方式，改为把配置中的 `engine.auto_start` 设为 `false`，并把 `engine.server_urls` 填成该服务地址。填写多个地址时，请求会随机分配并自动故障转移。

## 10. 使用 Open WebUI 测试（可选）

包内提供两个服务入口，区别很重要：`ocrcpm-server` 暴露的是**裁块级**的 OCR 模型，把整页图片直接发给它会绕过版面检测和 Markdown 组装；`ocrcpm-pipeline-server` 才是把整条流水线包成一个模型对外提供。文档前端应该连后者。

先生成服务模式配置：

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

然后分别打开终端启动 OCR 服务和 Pipeline 服务：

```bash
# 终端 1：OCR 服务
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
ocrcpm-server \
  --model-path "$PWD/ocr_model" \
  --host 127.0.0.1 \
  --port 18599
```

```bash
# 终端 2：Pipeline 服务
ocrcpm-pipeline-server \
  --config "$PWD/ocrcpm_config.yaml" \
  --host 0.0.0.0 \
  --port 18600 \
  --run-root "$PWD/openwebui_runs"
```

用下面的命令检查服务，两个接口应分别返回 OCR 模型和 `ocrcpm-pipeline`：

```bash
curl http://127.0.0.1:18599/v1/models
curl http://127.0.0.1:18600/v1/models
```

Open WebUI 建议使用单独的环境：

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

如果 Open WebUI 与 OCRCPM 不在同一台机器，请将 `OPENAI_API_BASE_URL` 中的 `127.0.0.1` 替换为 OCRCPM 服务器 IP。必须连接 Pipeline 服务的 `18600/v1`，不能直接连接 `18599/v1`。

浏览器访问 `http://<Open WebUI机器IP>:3000`，新建聊天并选择 `ocrcpm-pipeline`，上传测试图片，输入 `OCR` 后发送。识别结果会以 Markdown 返回，同时支持普通 JSON 和 OpenAI 兼容的 SSE 流式响应，运行文件保存在 `~/ocrcpm_demo/openwebui_runs/`。

## 11. 常见问题

### 找不到版面模型或 OCR 模型

```text
layout.model_dir not found
engine.model_path not found
```

检查对应目录下的 `config.json` 是否存在：

```bash
ls "$PWD/layout_model/config.json"
ls "$PWD/ocr_model/config.json"
```

### CUDA 不可用

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

如果输出为 `False`，需要安装与机器 CUDA 环境匹配的 PyTorch。也可以在初始化时改用 `--transformers-device cpu`，但会非常慢，并占用较多内存。

### llama-server 无法启动

先重新确认 CUDA、CMake 编译结果和服务日志：

```bash
nvidia-smi
cmake --build "$PWD/llama.cpp/build" --config Release -j
test -x "$PWD/llama.cpp/build/bin/llama-server"
```

如果端口被占用，可在初始化命令中更换 `--port`。详细启动错误位于运行目录的 `logs/engine.log`。

### Ollama 模型没有注册成功

```bash
OLLAMA_HOST=127.0.0.1:11434 ollama list
```

列表中应能看到 `--model-name` 指定的模型。如果没有，检查运行目录的 `logs/engine.log` 中 `ollama create` 的输出，并确认两个 GGUF 文件路径正确、投影仪文件确实是 `mmproj`。

如果 `ollama serve` 无法启动，通常是 11434 端口已被系统上已有的 Ollama 服务占用。可以更换 `--port`，或者直接复用已有服务：把配置中的 `engine.auto_start` 改为 `false`，并把 `engine.server_urls` 填成该服务地址。

### SGLang 或 vLLM 启动时报模型架构不支持

多半是环境里装的是官方版本，而不是第 8、9 节的适配分支。vLLM 的报错形如：

```text
ValueError: Currently, MiniCPMV only supports versions ...
```

按以下命令确认装的是哪一份代码：

```bash
"$PWD/sglang_env/bin/python" -m pip show sglang | grep -E "Version|Editable"
"$PWD/vllm_env/bin/python" -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)"
```

SGLang 应出现 `Editable project location` 并指向克隆下来的 `sglang/python`；vLLM 应指向克隆下来的 `vllm/vllm` 目录，而不是 `site-packages` 下的普通安装。如果 `sglang/` 或 `vllm/` 目录在安装后被移动或删除，也会出现这个问题。执行 vLLM 那条命令时不要停在 `vllm/` 目录里，否则 `import vllm` 会命中当前目录、给出误导性的结果。

这两个后端还要求模型目录 `config.json` 里的 `version` 字段与所用分支的预期一致，两边的适配都是按这个字段分派模型实现的。

### transformers 5.x 相关的加载报错

模型目录中随权重分发的胶水代码需要与所装的 transformers 版本匹配。已知两处：`processing_minicpmv.py` 如果没有透传 `downsample_mode`、或者没有按新签名调用 `get_slice_image_placeholder`，图像占位符数量会被算成 0，transformers 后端每一块都会失败；`modeling_minicpmv.py` 中取 `tokenizer.bos_id` 会在 transformers 5.x 上抛 `TokenizersBackend has no attribute bos_id`，需要回退到 `bos_token_id`。

这类问题只在 transformers 后端出现，因为 vLLM 和 SGLang 用的是各自的模型实现，不会执行到这两个函数。

### 已经生成过错误配置，或者换了后端却还在走旧后端

所有 `ocrcpm init` 写的都是同一个 `~/.config/ocrcpm/config.yaml`，重新执行对应那节的命令即可覆盖。想确认当前生效的是哪个后端，看这个文件里的 `engine.type`：

| `engine.type` | 对应后端 |
| --- | --- |
| `minicpm_transformers_local` | transformers-local（第 4 节） |
| `minicpm_llamacpp_openai_api` | llama.cpp（第 6 节） |
| `minicpm_ollama_api` | Ollama（第 7 节） |
| `minicpm_sglang_openai_api` | SGLang（第 8 节） |
| `minicpm_vllm_openai_api` | vLLM（第 9 节）或 http-client 模式 |

第 10 节的服务模式配置写在单独的 `ocrcpm_config.yaml` 里，不会影响默认配置。

## 12. Python API

除命令行外，流水线也可以直接在 Python 中分阶段调用：

```python
from ocrcpm import load_config, parse_layout_run, prepare_run, infer_run

cfg = load_config("my_config.yaml")
parse_layout_run(cfg)
prepare_run(cfg)
print(infer_run(cfg))
```
