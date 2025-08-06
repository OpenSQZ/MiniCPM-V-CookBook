# MiniCPM-V 4.0 - Ollama

## 1. 安装 Ollama

### macOS

[Download](https://ollama.com/download/Ollama.dmg)

### Windows

[Download](https://ollama.com/download/OllamaSetup.exe)

### Linux

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

[Manual install instructions](https://github.com/ollama/ollama/blob/main/docs/linux.md)

### Docker

The official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` is available on Docker Hub.

## 2. 快速使用

Ollama 可以直接使用:

```shell
ollama run openbmb/minicpm-v4
```

### 命令行
用空格分割输入问题，图片路径
```bash
这张图片描述了什么？ xx.jpg
```
### API
```python
with open(image_path, 'rb') as image_file:
    # 将图片文件转换为 base64 编码
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    data = {
    "model": "minicpm-v4",
    "prompt": query,
    "stream": False,
    "images": [encoded_string]# 列表可以放多张图，每张图用上面的方式转化为base64的格式
    }

    # 设置请求 URL
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=data)

    return response
```

## 3. 自定义方式

**若上述方式无法运行，请参考以下教程。**

### 环境需求

- [go](https://go.dev/doc/install) version 1.22 or above
- cmake version 3.24 or above
- C/C++ Compiler e.g. Clang on macOS, [TDM-GCC](https://github.com/jmeubank/tdm-gcc/releases) (Windows amd64) or [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (Windows arm64), GCC/Clang on Linux.

### 获取 GGUF 模型

*   HuggingFace: https://huggingface.co/openbmb/MiniCPM-V-4-gguf
*   魔搭社区: https://modelscope.cn/models/OpenBMB/MiniCPM-V-4-gguf

### 获取 OpenBMB 官方 Ollama 分支

```sh
git clone https://github.com/OpenBMB/ollama.git
cd ollama
```

### 构建 Ollama

```sh
cmake -B build
cmake --build build
```

### 启动 Ollama 服务

```sh
go run . serve
```

### 创建 ModelFile

编辑 ModelFile:

```sh
vim minicpmv4.Modelfile
```

ModelFile 的内容如下:

```plaintext
FROM ./MiniCPM-V-4/model/ggml-model-Q4_K_M.gguf
FROM ./MiniCPM-V-4/mmproj-model-f16.gguf

TEMPLATE """{{- if .Messages }}{{- range $i, $_ := .Messages }}{{- $last := eq (len (slice $.Messages $i)) 1 -}}<|im_start|>{{ .Role }}{{ .Content }}{{- if $last }}{{- if (ne .Role "assistant") }}<|im_end|><|im_start|>assistant{{ end }}{{- else }}<|im_end|>{{ end }}{{- end }}{{- else }}{{- if .System }}<|im_start|>system{{ .System }}<|im_end|>{{ end }}{{ if .Prompt }}<|im_start|>user{{ .Prompt }}<|im_end|>{{ end }}<|im_start|>assistant{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""

SYSTEM """You are a helpful assistant."""

PARAMETER top_p 0.8
PARAMETER num_ctx 4096
PARAMETER stop ["<|im_start|>","<|im_end|>"]
PARAMETER temperature 0.7
```
参数说明:

| first from | second from | num_ctx |
|-----|-----|-----|
| Your language GGUF model path | Your vision GGUF model path | Max Model length |

### 创建 Ollama 模型实例：
```bash
ollama create minicpm-v4 -f minicpmv4.Modelfile
```

### 另起一个命令行窗口，运行 Ollama 模型实例：
```bash
ollama run minicpm-v4
```

### 输入问题和图片 URL，以空格分隔
```bash
这张图片描述了什么？ xx.jpg
```
