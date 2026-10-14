# MiniCPM-V 4.7 — Ollama

> [!NOTE]
> Ollama 的模型代码已识别 MiniCPM-V 4.7（`convert_minicpmv.go`），但**其内置的 `llama.cpp`
> 必须先同步到 [PR #29416](https://github.com/ggml-org/llama.cpp/pull/29416)** 才能加载 4.7 的 GGUF
> （canvas M-RoPE 新格式引入了 `rope.section_order` 与 mmproj 张量 `v.tok_embd_sep`）。
> 在此之前，对 4.7 GGUF 执行 `ollama create` / `ollama run` 会失败。
> 可先用较新的 `llama.cpp`，或等待内置引擎更新后的 Ollama 正式版。

## 1. 安装 Ollama

- **macOS**：<https://ollama.com/download/Ollama.dmg>
- **Windows**：<https://ollama.com/download/OllamaSetup.exe>
- **Linux**：`curl -fsSL https://ollama.com/install.sh | sh`
- **Docker**：Docker Hub 上的 `ollama/ollama`

## 2. 快速开始（注册表）

待内置引擎更新的正式版发布后，从 Ollama 注册表拉取官方模型
（<https://ollama.com/openbmb/minicpm-v4.7>）：

```bash
ollama run openbmb/minicpm-v4.7
```

命令行中，提示词与图片路径用空格分隔：

```text
What is in the picture? xx.jpg
```

API：

```python
import base64, requests

with open("xx.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

data = {
    "model": "openbmb/minicpm-v4.7",
    "prompt": "What is in the picture?",
    "stream": False,
    "images": [encoded_string],  # 可放多张 base64 图片
}
response = requests.post("http://localhost:11434/api/generate", json=data)
```

## 3. 自定义模型（本地 GGUF）

**要加载本地 GGUF（如自定义量化或指定尺寸），创建一个 Modelfile。**

### 下载 GGUF

- HuggingFace：
  - 1B — <https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf>
  - 35B-A3B — <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf>
- ModelScope — `OpenBMB/MiniCPM-V-4.7-1B-gguf` / `OpenBMB/MiniCPM-V-4.7-35B-A3B-gguf`

### 创建 Modelfile

```bash
vim minicpmv4.7.Modelfile
```

Modelfile 内容（1B 示例）：

```plaintext
FROM ./MiniCPM-V-4.7-1B-gguf/MiniCPM-V-4_7-1B-Q4_K_M.gguf
FROM ./MiniCPM-V-4.7-1B-gguf/mmproj-model-f16.gguf

TEMPLATE """{{- if .Messages }}{{- range $i, $_ := .Messages }}{{- $last := eq (len (slice $.Messages $i)) 1 -}}<|im_start|>{{ .Role }}{{ .Content }}{{- if $last }}{{- if (ne .Role "assistant") }}<|im_end|><|im_start|>assistant{{ end }}{{- else }}<|im_end|>{{ end }}{{- end }}{{- else }}{{- if .System }}<|im_start|>system{{ .System }}<|im_end|>{{ end }}{{ if .Prompt }}<|im_start|>user{{ .Prompt }}<|im_end|>{{ end }}<|im_start|>assistant{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""

SYSTEM """You are a helpful assistant."""

PARAMETER top_p 0.8
PARAMETER num_ctx 8192
PARAMETER stop ["<|im_start|>", "<|im_end|>"]
PARAMETER temperature 0.7
```

| 第一个 FROM | 第二个 FROM | num_ctx |
| :--- | :--- | :--- |
| 语言模型 GGUF 路径 | 视觉投影 GGUF 路径 | 最大上下文长度 |

> **35B-A3B** 时，把第一个 `FROM` 改为 `MiniCPM-V-4.7-35B-A3B-gguf/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf`。

### 创建并运行

```bash
ollama create minicpm-v4.7 -f minicpmv4.7.Modelfile
ollama run minicpm-v4.7
```

命令行中提示词与图片路径用空格分隔：

```text
What is in the picture? xx.jpg
```

## 说明

- **模板**：MiniCPM-V 4.7 的 `chat_template.jinja` 与 4.6 逐字节相同，故上面的 `TEMPLATE` 与
  `openbmb/minicpm-v4.6` 所用一致。
- **不设 `RENDERER`**：与官方 `openbmb/minicpm-v4.6` 一致（family `qwen35`、无 renderer 字段），
  Ollama 走 native 模式，使用 GGUF 内嵌对话模板 + mmproj。
- **思考**：模板默认开启；如需直答，可通过请求传 `enable_thinking: false`
  （或改用支持 `chat_template_kwargs` 的运行时）。

## 推荐采样参数

在 instruct 模式下生成时，推荐使用：`temperature=0.7`、`top_p=0.95`、`top_k=100`、`min_p=0.0`、`presence_penalty=0`、`repetition_penalty=1.03`。
