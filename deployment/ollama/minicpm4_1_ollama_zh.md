# MiniCPM 4.1 - Ollama

> [!NOTE]
> MiniCPM 4.1 已上架 [OpenBMB 官方 Ollama 仓库](https://ollama.com/openbmb/minicpm4.1)。使用 Ollama `>= 0.30` 获得对 MiniCPM 4 / 4.1 的一等支持。

## 1. 安装 Ollama

- **macOS**：<https://ollama.com/download/Ollama.dmg>
- **Windows**：<https://ollama.com/download/OllamaSetup.exe>
- **Linux**：`curl -fsSL https://ollama.com/install.sh | sh`
- **Docker**：<https://hub.docker.com/r/ollama/ollama>

```bash
ollama --version   # 应 >= 0.30
```

## 2. 快速开始

```bash
ollama run openbmb/minicpm4.1
```

默认拉取 Q4 量化版本。指定具体量化：

```bash
ollama run openbmb/minicpm4.1:q4_K_M
ollama run openbmb/minicpm4.1:q5_K_M
ollama run openbmb/minicpm4.1:q8_0
```

## 3. 调用 HTTP API

Ollama 在 `http://localhost:11434` 暴露 REST 接口：

```python
import requests

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "openbmb/minicpm4.1",
        "messages": [{"role": "user", "content": "写一篇关于端侧 AI 的短文。"}],
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.8},
    },
)
print(resp.json()["message"]["content"])
```

也可以直接走 OpenAI 兼容接口 `/v1/chat/completions`，把 `openai` 客户端的 `base_url` 指向 `http://localhost:11434/v1` 即可。

## 4. 使用本地 GGUF 自定义

如果需要使用自定义量化（如 F16 原版或自己量化的 Q3_K_S），把 GGUF 写到 Modelfile 里：

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

## 5. 注意事项

- Ollama 暂未暴露 chat-template kwargs，所以无法通过 Ollama 单请求切换 `enable_thinking`。需要混合思考请使用 [vLLM](vllm.html)、[SGLang](sglang.html) 或 [llama.cpp](llamacpp.html) 指南。
