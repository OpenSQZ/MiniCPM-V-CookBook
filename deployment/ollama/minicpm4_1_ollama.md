# MiniCPM 4.1 - Ollama

> [!NOTE]
> MiniCPM 4.1 is on the [official OpenBMB Ollama registry](https://ollama.com/openbmb/minicpm4.1). Use Ollama `>= 0.30` for first-class MiniCPM 4 / 4.1 support.

## 1. Install Ollama

- **macOS**: <https://ollama.com/download/Ollama.dmg>
- **Windows**: <https://ollama.com/download/OllamaSetup.exe>
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
- **Docker**: <https://hub.docker.com/r/ollama/ollama>

```bash
ollama --version   # should be >= 0.30
```

## 2. Quick start

```bash
ollama run openbmb/minicpm4.1
```

By default the Q4 quantization is pulled. To force a specific quant:

```bash
ollama run openbmb/minicpm4.1:q4_K_M
ollama run openbmb/minicpm4.1:q5_K_M
ollama run openbmb/minicpm4.1:q8_0
```

## 3. Use the HTTP API

Ollama exposes a REST endpoint on `http://localhost:11434`:

```python
import requests

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "openbmb/minicpm4.1",
        "messages": [{"role": "user", "content": "Write a short article about edge AI."}],
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.8},
    },
)
print(resp.json()["message"]["content"])
```

OpenAI-compatible endpoint (`/v1/chat/completions`) also works — just point your `openai` client at `http://localhost:11434/v1`.

## 4. Customise with a local GGUF

If you want to use a custom quantization (e.g. the F16 master or your own Q3_K_S), drop the GGUF into a Modelfile:

```text
FROM ./MiniCPM4.1-8B-Q4_K_M.gguf
TEMPLATE """{{- if .Messages }}{{- range $i, $_ := .Messages }}{{- $last := eq (len (slice $.Messages $i)) 1 -}}<|im_start|>{{ .Role }}{{ .Content }}{{- if $last }}{{- if (ne .Role "assistant") }}<|im_end|><|im_start|>assistant{{ end }}{{- else }}<|im_end|>{{ end }}{{- end }}{{- end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""
SYSTEM """You are a helpful assistant."""
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER num_ctx 8192
PARAMETER stop ["<|im_start|>", "<|im_end|>"]
```

```bash
ollama create my-minicpm4.1 -f Modelfile
ollama run my-minicpm4.1
```

## 5. Notes

- Ollama doesn't expose chat-template kwargs, so `enable_thinking` cannot be toggled per request through Ollama yet. For hybrid reasoning use the [vLLM](vllm.html), [SGLang](sglang.html), or [llama.cpp](llamacpp.html) guide instead.
