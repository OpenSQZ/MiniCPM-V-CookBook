# MiniCPM-V 4.0 - Ollama

## 1. Install Ollama

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

## 2. Quick Start

The MiniCPM-V 4 model can be used directly:

```shell
ollama run openbmb/minicpm-v4
```

### Command Line
Separate the input prompt and the image path with space.
```bash
What is in the picture? xx.jpg
```
### API
```python
with open(image_path, 'rb') as image_file:
    # Convert the image file to a base64 encoded string
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    data = {
    "model": "minicpm-v4",
    "prompt": query,
    "stream": False,
    "images": [encoded_string] # The 'images' list can hold multiple base64-encoded images.
    }

    # Set request URL
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=data)

    return response
```

## 3. Customize model

**If the method above fails, please refer to the following guide.**

### Environment requirements

- [go](https://go.dev/doc/install) version 1.22 or above
- cmake version 3.24 or above
- C/C++ Compiler e.g. Clang on macOS, [TDM-GCC](https://github.com/jmeubank/tdm-gcc/releases) (Windows amd64) or [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) (Windows arm64), GCC/Clang on Linux.

### Download GGUF Model

*   HuggingFace: https://huggingface.co/openbmb/MiniCPM-V-4-gguf
*   ModelScope: https://modelscope.cn/models/OpenBMB/MiniCPM-V-4-gguf

### Clone Official OpenBMB Ollama Fork

```sh
git clone https://github.com/OpenBMB/ollama.git
cd ollama
```

### Configure and build Ollama

```sh
cmake -B build
cmake --build build
```

### Start Ollama Service

```sh
go run . serve
```

### Create a ModelFile

Create and edit a ModelFile:

```sh
vim minicpmv4.Modelfile
```

The content of the Modelfile should be as follows:

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
Parameter Descriptions:

| first from | second from | num_ctx |
|-----|-----|-----|
| Your language GGUF model path | Your vision GGUF model path | Max Model length |

### Create Ollama Model
```bash
ollama create minicpm-v4 -f minicpmv4.Modelfile
```

### Run
In a new terminal window, run the model instance:
```bash
ollama run minicpm-v4
```

### Input Prompt
Enter the prompt and the image path, separated by a space.
```bash
What is in the picture? xx.jpg
```
