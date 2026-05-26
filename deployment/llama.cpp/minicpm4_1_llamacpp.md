# MiniCPM 4.1 - llama.cpp

> [!NOTE]
> MiniCPM 4.1 GGUF weights are published officially at [`openbmb/MiniCPM4.1-8B-GGUF`](https://huggingface.co/openbmb/MiniCPM4.1-8B-GGUF). MiniCPM 4.1 is text-only, so the standard `llama-cli` is enough — no `--mmproj` projector needed.

## 1. Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# CPU / Metal
cmake -B build
cmake --build build --config Release

# CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

## 2. Get the GGUF

Download from HuggingFace or ModelScope and pick the quant that fits your memory budget:

- HuggingFace: <https://huggingface.co/openbmb/MiniCPM4.1-8B-GGUF>
- ModelScope: <https://modelscope.cn/models/OpenBMB/MiniCPM4.1-8B-GGUF>

Common picks:

| Quant | Size | Typical use |
| :--- | :--- | :--- |
| `Q4_K_M` | ~5 GB | Best speed / quality tradeoff |
| `Q5_K_M` | ~5.7 GB | Higher quality, marginal speed cost |
| `Q8_0` | ~8.5 GB | Near-FP16 quality |

## 3. Run

### One-shot prompt

```bash
./build/bin/llama-cli \
    -m /path/to/MiniCPM4.1-8B-Q4_K_M.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 \
    -p "Write a short article about edge AI."
```

### Interactive chat

```bash
./build/bin/llama-cli \
    -m /path/to/MiniCPM4.1-8B-Q4_K_M.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 \
    --conversation
```

### Hybrid reasoning

MiniCPM 4.1's chat template exposes `enable_thinking`. To turn reasoning on, pass `--jinja` and the chat-template kwargs through `--chat-template-kwargs`:

```bash
./build/bin/llama-cli \
    -m /path/to/MiniCPM4.1-8B-Q4_K_M.gguf \
    -c 8192 --temp 0.6 --top-p 0.95 \
    --jinja --chat-template-kwargs '{"enable_thinking": true}' \
    -p "If a train leaves at 9:00 traveling at 80 km/h to a city 200 km away, when does it arrive?"
```

> [!TIP]
> `enable_thinking` is `false` by default. Recommended sampling: `temperature=0.6, top_p=0.95` when on; `0.7, 0.8` when off.

## 4. Convert from PyTorch (optional)

```bash
python ./convert_hf_to_gguf.py /path/to/MiniCPM4.1-8B \
    --outfile /path/to/MiniCPM4.1-8B-F16.gguf --outtype f16

./build/bin/llama-quantize \
    /path/to/MiniCPM4.1-8B-F16.gguf \
    /path/to/MiniCPM4.1-8B-Q4_K_M.gguf \
    Q4_K_M
```

## 5. Argument reference

| Argument | Description |
| :--- | :--- |
| `-m, --model` | Path to the GGUF |
| `-p, --prompt` | One-shot prompt |
| `-c, --ctx-size` | Maximum context (up to 128K with InfLLM-V2 enabled in supported builds) |
| `--conversation` | Multi-turn interactive mode |
| `--jinja` | Use the model's Jinja chat template |
| `--chat-template-kwargs` | JSON kwargs forwarded to the chat template — used to flip `enable_thinking` |
