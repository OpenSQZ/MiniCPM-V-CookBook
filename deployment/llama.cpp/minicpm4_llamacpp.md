# MiniCPM 4 - llama.cpp

> [!NOTE]
> Official GGUF weights are available for both sizes:
> - 8B: [`openbmb/MiniCPM4-8B-GGUF`](https://huggingface.co/openbmb/MiniCPM4-8B-GGUF)
> - 0.5B: [`openbmb/MiniCPM4-0.5B-GGUF`](https://huggingface.co/openbmb/MiniCPM4-0.5B-GGUF)

## 1. Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON   # or omit -DGGML_CUDA=ON for CPU / Metal
cmake --build build --config Release
```

## 2. Run

### One-shot

```bash
./build/bin/llama-cli \
    -m /path/to/MiniCPM4-8B-Q4_K_M.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 \
    -p "Write an article about Artificial Intelligence."
```

### Interactive

```bash
./build/bin/llama-cli \
    -m /path/to/MiniCPM4-8B-Q4_K_M.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 --conversation
```

### 0.5B (edge-friendly)

The 0.5B variant runs comfortably on a laptop or a low-end edge device. Use a `Q4_K_M` GGUF for the most aggressive memory savings while keeping output quality acceptable.

## 3. Convert from PyTorch (optional)

```bash
python ./convert_hf_to_gguf.py /path/to/MiniCPM4-8B \
    --outfile /path/to/MiniCPM4-8B-F16.gguf --outtype f16

./build/bin/llama-quantize \
    /path/to/MiniCPM4-8B-F16.gguf \
    /path/to/MiniCPM4-8B-Q4_K_M.gguf \
    Q4_K_M
```

## 4. Notes

- MiniCPM 4 does **not** support `enable_thinking`. For hybrid reasoning use [MiniCPM 4.1](../minicpm4_1/deployment/llamacpp.html).
- The QAT variant `MiniCPM4-0.5B-QAT-Int4-GPTQ-format` is GPTQ-format (not GGUF) — use it via vLLM / SGLang instead.
