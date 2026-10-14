# MiniCPM-V 4.7 — Ollama

> [!NOTE]
> Ollama's model code already recognizes MiniCPM-V 4.7 (`convert_minicpmv.go`), but the
> **bundled `llama.cpp` must catch up to [PR #29416](https://github.com/ggml-org/llama.cpp/pull/29416)**
> before it can load 4.7 GGUFs (the canvas M-RoPE format adds `rope.section_order` and the
> `v.tok_embd_sep` mmproj tensor). Until then, `ollama create` / `ollama run` on a 4.7 GGUF will
> fail. Use this guide together with a recent `llama.cpp` or wait for the Ollama release that
> bundles the updated engine.

## 1. Install Ollama

- **macOS**: <https://ollama.com/download/Ollama.dmg>
- **Windows**: <https://ollama.com/download/OllamaSetup.exe>
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
- **Docker**: `ollama/ollama` on Docker Hub

## 2. Quick Start (registry)

Once the release that bundles the updated engine is out, pull the official model from the Ollama
registry (<https://ollama.com/openbmb/minicpm-v4.7>):

```bash
ollama run openbmb/minicpm-v4.7
```

Command line: separate the prompt and the image path with a space:

```text
What is in the picture? xx.jpg
```

API:

```python
import base64, requests

with open("xx.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

data = {
    "model": "openbmb/minicpm-v4.7",
    "prompt": "What is in the picture?",
    "stream": False,
    "images": [encoded_string],  # list can hold multiple base64-encoded images
}
response = requests.post("http://localhost:11434/api/generate", json=data)
```

## 3. Customize model (local GGUF)

**To load a local GGUF (e.g. a custom quantization or a specific size), create a Modelfile.**

### Download GGUF Model

- HuggingFace:
  - 1B — <https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf>
  - 35B-A3B — <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf>
- ModelScope — `OpenBMB/MiniCPM-V-4.7-1B-gguf` / `OpenBMB/MiniCPM-V-4.7-35B-A3B-gguf`

### Create a Modelfile

```bash
vim minicpmv4.7.Modelfile
```

Modelfile content (1B example):

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

| first FROM | second FROM | num_ctx |
| :--- | :--- | :--- |
| Path to the language GGUF | Path to the vision projector GGUF | Maximum context length |

> For **35B-A3B**, change the first `FROM` to `MiniCPM-V-4.7-35B-A3B-gguf/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf`.

### Create and run the Ollama model

```bash
ollama create minicpm-v4.7 -f minicpmv4.7.Modelfile
ollama run minicpm-v4.7
```

Command line: prompt and image path separated by a space:

```text
What is in the picture? xx.jpg
```

## Notes

- **Template**: MiniCPM-V 4.7's `chat_template.jinja` is byte-identical to 4.6's, so the `TEMPLATE`
  above is the same one used for `openbmb/minicpm-v4.6`.
- **No `RENDERER` directive**: like the official `openbmb/minicpm-v4.6` model (family `qwen35`,
  no renderer field), Ollama runs in native mode using the GGUF-embedded chat template + mmproj.
- **Thinking**: enabled by default in the template; to get plain replies, add
  `PARAMETER` support for `enable_thinking: false` (via the request) or serve the GGUF through a
  runtime that lets you pass `chat_template_kwargs`.

## Recommended sampling parameters

For instruct-mode generation we recommend: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.
