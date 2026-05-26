# MiniCPM-V 4.6 - Ollama

> [!NOTE]
> MiniCPM-V 4.6 is officially supported by Ollama starting from **v0.30**. Please make sure your Ollama version is **0.30 or newer**; older releases do not include the model and will fail to load it.

## 1. Install Ollama

- **macOS**: download from <https://ollama.com/download/Ollama.dmg>.
- **Windows**: download from <https://ollama.com/download/OllamaSetup.exe>.
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`, or follow the [Linux install guide](https://github.com/ollama/ollama/blob/main/docs/linux.md).
- **Docker**: official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` is on Docker Hub.

Verify your version is at least 0.30:

```bash
ollama --version
```

## 2. Quick Start

Pull and run the official OpenBMB model from the Ollama registry (<https://ollama.com/openbmb/minicpm-v4.6>):

```bash
ollama run openbmb/minicpm-v4.6
# or, for the Thinking variant
ollama run openbmb/minicpm-v4.6-thinking
```

### Command line

Separate the input prompt and the image path with a space:

```text
What is in the picture? xx.jpg
```

### API

```python
import base64, requests

with open(image_path, 'rb') as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    data = {
        "model": "openbmb/minicpm-v4.6",
        "prompt": query,
        "stream": False,
        "images": [encoded_string],  # the list can hold multiple base64-encoded images
    }
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=data)
```

## 3. Customize model

**If you'd like to load a local GGUF (e.g. a custom quantization), follow the steps below.**

### Download GGUF Model

- HuggingFace: <https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf>
- ModelScope: <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-gguf>

(Or `openbmb/MiniCPM-V-4.6-Thinking-gguf` for the Thinking variant.)

### Create a ModelFile

```bash
vim minicpmv4.6.Modelfile
```

ModelFile content:

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

| first FROM | second FROM | num_ctx |
| :--- | :--- | :--- |
| Path to the language GGUF | Path to the vision projector GGUF | Maximum context length |

### Create the Ollama model

```bash
ollama create minicpm-v4.6 -f minicpmv4.6.Modelfile
```

### Run

In a new terminal:

```bash
ollama run minicpm-v4.6
```

### Input prompt

Enter the prompt and the image path separated by a space:

```text
What is in the picture? xx.jpg
```
