# MiniCPM 4 - Ollama

> [!NOTE]
> MiniCPM 4 已上架 [OpenBMB 官方 Ollama 仓库](https://ollama.com/openbmb/minicpm4)。需要 Ollama `>= 0.30`。

## 1. 安装 Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS — https://ollama.com/download/Ollama.dmg
# Windows — https://ollama.com/download/OllamaSetup.exe

ollama --version   # >= 0.30
```

## 2. 快速开始

```bash
# 8B（默认 Q4）
ollama run openbmb/minicpm4

# 指定量化
ollama run openbmb/minicpm4:q5_K_M
ollama run openbmb/minicpm4:q8_0
```

## 3. 调用 API

```python
import requests
resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "openbmb/minicpm4",
        "messages": [{"role": "user", "content": "写一篇关于人工智能的文章。"}],
        "stream": False,
    },
)
print(resp.json()["message"]["content"])
```

OpenAI 兼容接口在 `http://localhost:11434/v1`。

## 4. 注意事项

- MiniCPM 4 **不支持**混合思考。需要 `enable_thinking` 请使用 [MiniCPM 4.1](../minicpm4_1/deployment/ollama.html)。
- 0.5B 尺寸目前没有官方 Ollama tag，需要基于 [`MiniCPM4-0.5B-GGUF`](https://huggingface.co/openbmb/MiniCPM4-0.5B-GGUF) 自行写 Modelfile。
