# MiniCPM 5 - Ollama

> [!NOTE]
> [Ollama](https://ollama.com) is the easiest CLI / daemon path to run MiniCPM 5 on a laptop or desktop — one binary, no Python, no CUDA toolkit. It consumes the same GGUF files we ship for `llama.cpp`. Once published, MiniCPM 5 will appear on the [official OpenBMB Ollama registry](https://ollama.com/openbmb).

## 1. Install Ollama

- **macOS**: <https://ollama.com/download/Ollama.dmg>
- **Windows**: <https://ollama.com/download/OllamaSetup.exe>
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
- **Docker**: <https://hub.docker.com/r/ollama/ollama>

```bash
ollama --version
```

## 2. Quick start (registry)

```bash
ollama run openbmb/minicpm5
```

To force a specific quant:

```bash
ollama run openbmb/minicpm5:q4_K_M
ollama run openbmb/minicpm5:q8_0
ollama run openbmb/minicpm5:fp16
```

## 3. Quick start (local GGUF)

If the registry image is unavailable in your network, build a Modelfile from the released GGUF directly:

```bash
brew install ollama                 # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh   (Linux)

OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &

mkdir -p ~/minicpm5-1b && cd ~/minicpm5-1b
huggingface-cli download openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q4_K_M.gguf --local-dir .

QUANT=Q4_K_M
cat > Modelfile <<EOF
FROM ./MiniCPM5-1B-${QUANT}.gguf

# MiniCPM 5 chat template (ChatML-style)
TEMPLATE """{{- if .Messages -}}
{{- range .Messages -}}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end -}}
<|im_start|>assistant
{{ end -}}"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"

# Defaults are tuned for No-Think mode
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER num_ctx 8192
EOF

ollama create minicpm5-1b -f Modelfile
ollama run minicpm5-1b
```

## 4. Choosing a quant

| Quant | Disk | RAM @ 8K ctx | Quality | Suggested tag |
| :--- | ---: | ---: | :--- | :--- |
| F16 | 2.1 GB | ~3 GB | reference | `:fp16` |
| Q8_0 | 1.1 GB | ~2 GB | ~indistinguishable from F16 | `:q8_0` |
| **Q4_K_M** | **657 MB** | **~1.3 GB** | small drop, ideal for laptops | **`:q4_K_M`** *(default)* |

## 5. API access

Ollama serves an OpenAI-compatible REST endpoint on `http://localhost:11434/v1`:

```bash
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "minicpm5-1b",
        "messages": [{"role": "user", "content": "用一句话解释 GQA。"}],
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 1024
    }'
```

Or use the Ollama-native API:

```bash
curl http://localhost:11434/api/chat -d '{
    "model": "minicpm5-1b",
    "messages": [{"role":"user","content":"1+1=?"}],
    "stream": false,
    "options": {"temperature": 0.7, "top_p": 0.95}
}'
```

## 6. Think vs No-Think

The Modelfile above defaults to **No-Think** (`temperature=0.7, top_p=0.95`). To switch a single conversation to **Think** mode, override the sampling params at request time:

```bash
ollama run minicpm5-1b --temperature 0.9 --top-p 0.95
```

Or bake it into a separate model tag:

```Modelfile
PARAMETER temperature 0.9
PARAMETER top_p 0.95
```

Then `ollama create minicpm5-1b-think -f Modelfile.think`.

> ℹ️ Ollama 0.24+ does **not** auto-evaluate the GGUF-embedded Jinja chat template; it falls back to the Modelfile's Go `TEMPLATE` block, which does not propagate `chat_template_kwargs`. To force the Think path with auto-injected `<think>\n`, use raw mode and prepend it manually:
>
> ```bash
> curl http://localhost:11434/api/generate -d '{
>     "model": "minicpm5-1b",
>     "raw": true,
>     "prompt": "<|im_start|>user\n鸡兔同笼…<|im_end|>\n<|im_start|>assistant\n<think>\n",
>     "options": {"temperature": 0.9, "top_p": 0.95}
> }'
> ```

## 7. Higher throughput on Apple Silicon

For long-context workloads, raise `num_ctx`:

```Modelfile
PARAMETER num_ctx 32768
```

For sustained throughput with larger KV caches, set `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0` in the environment before `ollama serve`.

## See also

- [`llama.cpp`](llamacpp.html) — the engine behind Ollama; full GGUF build pipeline
- [`mlx.html`](mlx.html) — alternative on-device path on Apple Silicon (faster for Q4)
