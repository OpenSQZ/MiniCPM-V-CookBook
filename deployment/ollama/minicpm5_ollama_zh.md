# MiniCPM 5 - Ollama

> [!NOTE]
> [Ollama](https://ollama.com) 是在笔记本 / 桌面上运行 MiniCPM 5 最简单的 CLI / 守护进程方案 —— 单个二进制、无 Python、无 CUDA toolkit。它直接消费我们为 `llama.cpp` 提供的 GGUF 文件。模型已发布在 [OpenBMB 官方 Ollama registry](https://ollama.com/openbmb/minicpm5)。

> [!IMPORTANT]
> **MiniCPM 5 的 tokenizer 上游支持还在路上**（[llama.cpp PR #23384](https://github.com/ggml-org/llama.cpp/pull/23384)）。稳定版 Ollama（基于当前上游 `llama.cpp` 编译）实际上能产出正确结果，因为它的 `llama-bpe` 预分词器对 MiniCPM 5 的两阶段正则在常见输入下功能等价。
>
> 如果你想走 forward-compatible 路径 —— 即正确识别未来标注 `tokenizer.ggml.pre = "minicpm5"` 的 GGUF、并走 Ollama 新 Go engine —— 请在上游合入前临时使用我们的 fork：
>
> ```bash
> git clone -b MiniCPM5 https://github.com/tc-mb/ollama.git
> cd ollama
> go build -o ollama .          # macOS Apple Silicon：内置 Metal
> ./ollama serve &
> ./ollama run openbmb/minicpm5
> ```
>
> 补丁全部在 `model/models/llama/model.go` 一个 Go 文件里，与上游 PR 完全一致。MiniCPM 5 不需要 vendored `llama.cpp` 改动。

## 1. 安装 Ollama

- **macOS**: <https://ollama.com/download/Ollama.dmg>
- **Windows**: <https://ollama.com/download/OllamaSetup.exe>
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
- **Docker**: <https://hub.docker.com/r/ollama/ollama>

```bash
ollama --version
```

## 2. 快速启动（registry）

```bash
ollama run openbmb/minicpm5
```

指定具体量化：

```bash
ollama run openbmb/minicpm5:q4_K_M
ollama run openbmb/minicpm5:q8_0
ollama run openbmb/minicpm5:fp16
```

## 3. 快速启动（本地 GGUF）

若 registry 镜像在你的网络下不可用，可直接基于发布的 GGUF 写 Modelfile：

```bash
brew install ollama                 # macOS
# 或: curl -fsSL https://ollama.com/install.sh | sh   （Linux）

OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &

mkdir -p ~/minicpm5-1b && cd ~/minicpm5-1b
huggingface-cli download openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q4_K_M.gguf --local-dir .

QUANT=Q4_K_M
cat > Modelfile <<EOF
FROM ./MiniCPM5-1B-${QUANT}.gguf

# MiniCPM 5 chat template（ChatML 风格）。
# 前缀的 <s> 是 **必须** 的——官方 GGUF 设置了 add_bos_token=False，
# 不显式加 BOS 模型会输出乱码（\`\`\`<think> ... 循环重复）。
TEMPLATE """<s>{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# 默认按 No-Think 模式调优
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER num_ctx 8192
EOF

ollama create minicpm5-1b -f Modelfile
ollama run minicpm5-1b
```

## 4. 量化档位选择

| 量化 | 磁盘 | RAM @ 8K ctx | 质量 | 推荐 tag |
| :--- | ---: | ---: | :--- | :--- |
| F16 | 2.1 GB | ~3 GB | 参考精度 | `:fp16` |
| Q8_0 | 1.1 GB | ~2 GB | 与 F16 几乎无差 | `:q8_0` |
| **Q4_K_M** | **657 MB** | **~1.3 GB** | 轻微下降，笔记本首选 | **`:q4_K_M`**（默认） |

## 5. API 访问

Ollama 暴露 OpenAI 兼容 REST 接口在 `http://localhost:11434/v1`：

```bash
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "minicpm5-1b",
        "messages": [{"role": "user", "content": "用一句话解释 GQA。"}],
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 1024
    }'
```

或者使用 Ollama 原生 API：

```bash
curl http://localhost:11434/api/chat -d '{
    "model": "minicpm5-1b",
    "messages": [{"role":"user","content":"1+1=?"}],
    "stream": false,
    "options": {"temperature": 0.7, "top_p": 0.95}
}'
```

## 6. Think / No-Think 切换

上面的 Modelfile 默认按 **No-Think** 模式调优（`temperature=0.7, top_p=0.95`）。需要单次切到 **Think** 模式时直接覆盖采样参数：

```bash
ollama run minicpm5-1b --temperature 0.9 --top-p 0.95
```

或者另存一个 Think 模型 tag：

```Modelfile
PARAMETER temperature 0.9
PARAMETER top_p 0.95
```

```bash
ollama create minicpm5-1b-think -f Modelfile.think
```

> ℹ️ Ollama 0.24+ **不会**自动执行 GGUF 内置的 Jinja chat template，而是回退到 Modelfile 中的 Go `TEMPLATE` 块 —— 这种回退不会传递 `chat_template_kwargs`。要让 Think 路径自动注入 `<think>\n`，建议使用 raw 模式手动前置：
>
> ```bash
> curl http://localhost:11434/api/generate -d '{
>     "model": "minicpm5-1b",
>     "raw": true,
>     "prompt": "<|im_start|>user\n鸡兔同笼…<|im_end|>\n<|im_start|>assistant\n<think>\n",
>     "options": {"temperature": 0.9, "top_p": 0.95}
> }'
> ```

## 7. Apple Silicon 上的更高吞吐

长上下文场景把 `num_ctx` 提上去：

```Modelfile
PARAMETER num_ctx 32768
```

为了在更大 KV cache 下保持稳定吞吐，启动 `ollama serve` 前在环境中加：`OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0`。

## 相关文档

- [`llama.cpp`](llamacpp.html) —— Ollama 背后的引擎，完整的 GGUF 构建流程
- [`mlx.html`](mlx.html) —— Apple Silicon 上更快的 4-bit 路径
