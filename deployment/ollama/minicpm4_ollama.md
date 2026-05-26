# MiniCPM 4 - Ollama

> [!NOTE]
> MiniCPM 4 is published on the [official OpenBMB Ollama registry](https://ollama.com/openbmb/minicpm4). Requires Ollama `>= 0.30`.

## 1. Install Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS — https://ollama.com/download/Ollama.dmg
# Windows — https://ollama.com/download/OllamaSetup.exe

ollama --version   # >= 0.30
```

## 2. Quick start

```bash
# 8B (default Q4)
ollama run openbmb/minicpm4

# Specific quant
ollama run openbmb/minicpm4:q5_K_M
ollama run openbmb/minicpm4:q8_0
```

## 3. Call the API

```python
import requests
resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "openbmb/minicpm4",
        "messages": [{"role": "user", "content": "Write an article about AI."}],
        "stream": False,
    },
)
print(resp.json()["message"]["content"])
```

OpenAI-compatible endpoint also available at `http://localhost:11434/v1`.

## 4. Notes

- MiniCPM 4 does **not** support hybrid reasoning. Use [MiniCPM 4.1](../minicpm4_1/deployment/ollama.html) if you need `enable_thinking`.
- For the 0.5B size, build a local Modelfile against the [`MiniCPM4-0.5B-GGUF`](https://huggingface.co/openbmb/MiniCPM4-0.5B-GGUF) weights (no official Ollama tag yet).
