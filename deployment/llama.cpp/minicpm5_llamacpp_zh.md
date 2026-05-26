# MiniCPM 5 - llama.cpp

> [!NOTE]
> MiniCPM 5 的 GGUF 官方权重托管在 [`openbmb/MiniCPM5-1B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF)。由于 MiniCPM 5 本身就是标准 `LlamaForCausalLM`，发布的 GGUF 可在**原版 `llama.cpp`** 以及任何基于 `llama.cpp` 的运行时（Ollama / `llama-cpp-python`）上直接运行 —— **无需打补丁**。
>
> `llama.cpp` 是 **CPU / 端侧 / 消费级 GPU** 部署的推荐路径，可在笔记本、单板机、Apple Silicon、Windows 上完全不依赖 Python 即可运行。

## 1. 官方 GGUF 文件清单

| 文件 | 大小 | 适用场景 |
| :--- | ---: | :--- |
| `MiniCPM5-1B-F16.gguf` | 2.1 GB | 参考精度，CPU/GPU 表现一致 |
| `MiniCPM5-1B-Q8_0.gguf` | 1.1 GB | 相对 F16 几乎无损，磁盘占用减半 |
| `MiniCPM5-1B-Q4_K_M.gguf` | 657 MB | 端侧 / 移动级硬件，最低显存 |

## 2. 构建 llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# CPU / Metal
cmake -B build
cmake --build build --config Release -j

# CUDA
# cmake -B build -DGGML_CUDA=ON
# cmake --build build --config Release -j
```

## 3. 快速体验

```bash
huggingface-cli download openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q4_K_M.gguf --local-dir ./minicpm5

# 交互式 chat（自动应用 chat template）
./build/bin/llama-cli \
    -m ./minicpm5/MiniCPM5-1B-Q4_K_M.gguf \
    -n 2048 --temp 0.7 --top-p 0.95 -ngl 99
```

## 4. OpenAI 兼容 server

```bash
./build/bin/llama-server \
    -m MiniCPM5-1B-Q4_K_M.gguf \
    --port 8080 -ngl 99 -c 8192 --jinja

curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "temperature": 0.7, "top_p": 0.95, "max_tokens": 256
    }'
```

### 启用混合思考

MiniCPM 5 的 chat template 暴露了 `enable_thinking` 变量。要打开思考模式，需同时传入 `--jinja` 与 `--chat-template-kwargs`：

```bash
./build/bin/llama-cli \
    -m MiniCPM5-1B-Q4_K_M.gguf \
    -c 8192 --temp 0.9 --top-p 0.95 \
    --jinja --chat-template-kwargs '{"enable_thinking": true}' \
    -p "鸡兔同笼，头共 10 个，脚共 28 只，问鸡和兔各几只？"
```

| 模式 | `--temp` | `--top-p` | 适用场景 |
| :--- | :---: | :---: | :--- |
| Think | 0.9 | 0.95 | 推理、数学、代码、多步任务 |
| No-Think | 0.7 | 0.95 | 快速对话、延迟敏感场景 |

## 5. 自有权重转 GGUF

若你训练了自己的 MiniCPM5-1B 变体（继续预训练、领域 SFT 等）想发布 GGUF：

```bash
# 在前面克隆好的 llama.cpp 目录中：
SRC=/path/to/your-MiniCPM5-fp16-hf
OUT=/path/to/output

python ./convert_hf_to_gguf.py "$SRC" --outfile "$OUT/F16.gguf" --outtype f16
./build/bin/llama-quantize "$OUT/F16.gguf" "$OUT/Q4_K_M.gguf" Q4_K_M
./build/bin/llama-quantize "$OUT/F16.gguf" "$OUT/Q8_0.gguf"   Q8_0
```

## 6. 参数速查

| 参数 | 说明 |
| :--- | :--- |
| `-m, --model` | GGUF 文件路径 |
| `-p, --prompt` | 一次性 prompt |
| `-c, --ctx-size` | 最大上下文（内存够时可拉到 131,072） |
| `--conversation` | 多轮交互模式 |
| `--jinja` | 使用模型自带的 Jinja chat template |
| `--chat-template-kwargs` | 转发给 chat template 的 JSON 参数 —— 用于切换 `enable_thinking` |
| `-ngl` | 卸载到 GPU 的层数（`99` 表示全部） |

## 相关文档

- [`ollama.html`](ollama.html) —— 直接从这些 GGUF 启动 `ollama run`
