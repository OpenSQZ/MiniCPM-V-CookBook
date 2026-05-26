# MiniCPM 5 - llama.cpp

> [!NOTE]
> MiniCPM 5 GGUF weights are published officially at [`openbmb/MiniCPM5-1B-GGUF`](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF). Because MiniCPM 5 is a vanilla `LlamaForCausalLM`, the released GGUF runs on **vanilla `llama.cpp`** and every `llama.cpp`-based runtime (Ollama / `llama-cpp-python`) — no patched build required.
>
> `llama.cpp` is the recommended path for **CPU / edge / consumer-GPU** deployment. The model fits on laptops, single-board computers, Apple Silicon, and Windows boxes with no Python at all.

## 1. Released GGUF artifacts

| File | Size | Use case |
| :--- | ---: | :--- |
| `MiniCPM5-1B-F16.gguf` | 2.1 GB | reference quality, uniform CPU/GPU performance |
| `MiniCPM5-1B-Q8_0.gguf` | 1.1 GB | very small quality drop vs F16, half the disk |
| `MiniCPM5-1B-Q4_K_M.gguf` | 657 MB | edge / mobile-class hardware, minimal VRAM |

## 2. Build llama.cpp

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

## 3. TL;DR — run a release GGUF

```bash
huggingface-cli download openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q4_K_M.gguf --local-dir ./minicpm5

# Interactive chat (auto-applies the chat template)
./build/bin/llama-cli \
    -m ./minicpm5/MiniCPM5-1B-Q4_K_M.gguf \
    -n 2048 --temp 0.7 --top-p 0.95 -ngl 99
```

## 4. OpenAI-compatible server

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

### Hybrid reasoning

MiniCPM 5's chat template exposes `enable_thinking`. To turn reasoning on, pass `--jinja` and the chat-template kwargs through `--chat-template-kwargs`:

```bash
./build/bin/llama-cli \
    -m MiniCPM5-1B-Q4_K_M.gguf \
    -c 8192 --temp 0.9 --top-p 0.95 \
    --jinja --chat-template-kwargs '{"enable_thinking": true}' \
    -p "鸡兔同笼，头共 10 个，脚共 28 只，问鸡和兔各几只？"
```

| Mode | `--temp` | `--top-p` | When to use |
| :--- | :---: | :---: | :--- |
| Think | 0.9 | 0.95 | reasoning, math, code, multi-step |
| No-Think | 0.7 | 0.95 | fast assistant, latency-bound |

## 5. Build a GGUF from your own checkpoint

If you've trained your own MiniCPM5-1B variant (continue-pretraining, domain SFT, …) and want to publish a GGUF:

```bash
# In the cloned llama.cpp repo:
SRC=/path/to/your-MiniCPM5-fp16-hf
OUT=/path/to/output

python ./convert_hf_to_gguf.py "$SRC" --outfile "$OUT/F16.gguf" --outtype f16
./build/bin/llama-quantize "$OUT/F16.gguf" "$OUT/Q4_K_M.gguf" Q4_K_M
./build/bin/llama-quantize "$OUT/F16.gguf" "$OUT/Q8_0.gguf"   Q8_0
```

## 6. Argument reference

| Argument | Description |
| :--- | :--- |
| `-m, --model` | Path to the GGUF |
| `-p, --prompt` | One-shot prompt |
| `-c, --ctx-size` | Maximum context (up to 131,072 with enough RAM) |
| `--conversation` | Multi-turn interactive mode |
| `--jinja` | Use the model's Jinja chat template |
| `--chat-template-kwargs` | JSON kwargs forwarded to the chat template — used to flip `enable_thinking` |
| `-ngl` | Number of GPU layers to offload (`99` = all) |

## See also

- [`ollama.html`](ollama.html) — `ollama run` directly from these GGUFs
