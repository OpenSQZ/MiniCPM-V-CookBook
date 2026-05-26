# MiniCPM 4 - llama.cpp

> [!NOTE]
> 两种尺寸均有官方 GGUF 权重：
> - 8B：[`openbmb/MiniCPM4-8B-GGUF`](https://huggingface.co/openbmb/MiniCPM4-8B-GGUF)
> - 0.5B：[`openbmb/MiniCPM4-0.5B-GGUF`](https://huggingface.co/openbmb/MiniCPM4-0.5B-GGUF)

## 1. 构建 llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON   # CPU / Metal 请去掉 -DGGML_CUDA=ON
cmake --build build --config Release
```

## 2. 运行

### 单次

```bash
./build/bin/llama-cli \
    -m /path/to/MiniCPM4-8B-Q4_K_M.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 \
    -p "写一篇关于人工智能的文章。"
```

### 交互式

```bash
./build/bin/llama-cli \
    -m /path/to/MiniCPM4-8B-Q4_K_M.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 --conversation
```

### 0.5B（端侧友好）

0.5B 变体可以在普通笔记本或低配端侧设备运行。要进一步压低内存占用，可选 `Q4_K_M` GGUF，质量损失可控。

## 3. 从 PyTorch 转换（可选）

```bash
python ./convert_hf_to_gguf.py /path/to/MiniCPM4-8B \
    --outfile /path/to/MiniCPM4-8B-F16.gguf --outtype f16

./build/bin/llama-quantize \
    /path/to/MiniCPM4-8B-F16.gguf \
    /path/to/MiniCPM4-8B-Q4_K_M.gguf \
    Q4_K_M
```

## 4. 注意事项

- MiniCPM 4 **不支持** `enable_thinking`。需要混合思考请使用 [MiniCPM 4.1](../minicpm4_1/deployment/llamacpp.html)。
- QAT 变体 `MiniCPM4-0.5B-QAT-Int4-GPTQ-format` 是 GPTQ 格式（不是 GGUF），通过 vLLM / SGLang 使用。
