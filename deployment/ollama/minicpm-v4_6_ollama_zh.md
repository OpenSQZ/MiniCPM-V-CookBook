# MiniCPM-V 4.6 - Ollama

> [!NOTE]
> Ollama 自 **v0.30** 起官方支持 MiniCPM-V 4.6。请确保 Ollama 版本 **不低于 0.30**，旧版本不包含该模型支持，会加载失败。

## 1. 安装 Ollama

- **macOS**：[下载](https://ollama.com/download/Ollama.dmg)
- **Windows**：[下载](https://ollama.com/download/OllamaSetup.exe)
- **Linux**：`curl -fsSL https://ollama.com/install.sh | sh`，或参照 Ollama 官方[手动安装指南](https://github.com/ollama/ollama/blob/main/docs/linux.md)。
- **Docker**：官方的 [Ollama Docker 镜像](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` 已在 Docker Hub 上提供。

确认版本不低于 0.30：

```bash
ollama --version
```

## 2. 快速使用

从 Ollama registry 拉取并运行 OpenBMB 官方模型（<https://ollama.com/openbmb/minicpm-v4.6>）：

```bash
ollama run openbmb/minicpm-v4.6
# 或 Thinking 版本
ollama run openbmb/minicpm-v4.6-thinking
```

### 命令行

用空格分隔输入问题和图片路径：

```text
这张图片描述了什么？ xx.jpg
```

### API

```python
import base64, requests

with open(image_path, 'rb') as image_file:
    # 将图片文件转换为 base64 编码
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    data = {
        "model": "openbmb/minicpm-v4.6",
        "prompt": query,
        "stream": False,
        "images": [encoded_string],  # 列表可以放多张图，每张图用上面的方式转化为 base64
    }
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=data)
```

## 3. 自定义方式

**如果需要加载本地 GGUF（例如自定义量化版本），请按下面的步骤操作。**

### 获取 GGUF 模型

- HuggingFace：<https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf>
- 魔搭社区：<https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-gguf>

（Thinking 版本：`openbmb/MiniCPM-V-4.6-Thinking-gguf`）

### 创建 ModelFile

```bash
vim minicpmv4.6.Modelfile
```

ModelFile 的内容如下：

```plaintext
FROM ./MiniCPM-V-4.6/MiniCPM-V-4.6-Q4_K_M.gguf
FROM ./MiniCPM-V-4.6/mmproj-MiniCPM-V-4.6-F16.gguf

TEMPLATE """{{- if .Messages }}{{- range $i, $_ := .Messages }}{{- $last := eq (len (slice $.Messages $i)) 1 -}}<|im_start|>{{ .Role }}{{ .Content }}{{- if $last }}{{- if (ne .Role "assistant") }}<|im_end|><|im_start|>assistant{{ end }}{{- else }}<|im_end|>{{ end }}{{- end }}{{- else }}{{- if .System }}<|im_start|>system{{ .System }}<|im_end|>{{ end }}{{ if .Prompt }}<|im_start|>user{{ .Prompt }}<|im_end|>{{ end }}<|im_start|>assistant{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""

SYSTEM """You are a helpful assistant."""

PARAMETER top_p 0.8
PARAMETER num_ctx 8192
PARAMETER stop ["<|im_start|>", "<|im_end|>"]
PARAMETER temperature 0.7
```

参数说明：

| first FROM | second FROM | num_ctx |
| :--- | :--- | :--- |
| 语言模型 GGUF 路径 | 视觉 projector GGUF 路径 | 最大上下文长度 |

### 创建 Ollama 模型实例

```bash
ollama create minicpm-v4.6 -f minicpmv4.6.Modelfile
```

### 另起一个命令行窗口，运行 Ollama 模型实例

```bash
ollama run minicpm-v4.6
```

### 输入问题和图片路径，以空格分隔

```text
这张图片描述了什么？ xx.jpg
```
