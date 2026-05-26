# MiniCPM 4.1 - llama.cpp

> [!NOTE]
> MiniCPM 4.1 官方 GGUF 权重在 [`openbmb/MiniCPM4.1-8B-GGUF`](https://huggingface.co/openbmb/MiniCPM4.1-8B-GGUF)。MiniCPM 4.1 为纯文本模型，使用标准 `llama-cli` 即可，无需 `--mmproj` projector。

## 1. 构建 llama.cpp

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

## 2. 获取 GGUF

从 HuggingFace 或 ModelScope 下载，按显存预算挑选量化版本：

- HuggingFace: <https://huggingface.co/openbmb/MiniCPM4.1-8B-GGUF>
- ModelScope: <https://modelscope.cn/models/OpenBMB/MiniCPM4.1-8B-GGUF>

常用选择：

| 量化 | 体积 | 典型用途 |
| :--- | :--- | :--- |
| `Q4_K_M` | 约 5 GB | 速度 / 质量最佳折衷 |
| `Q5_K_M` | 约 5.7 GB | 质量更高，速度损失很小 |
| `Q8_0` | 约 8.5 GB | 接近 FP16 质量 |

## 3. 运行

### 单条 prompt

```bash
./build/bin/llama-cli \
    -m /path/to/MiniCPM4.1-8B-Q4_K_M.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 \
    -p "写一篇关于端侧 AI 的短文。"
```

### 交互式对话

```bash
./build/bin/llama-cli \
    -m /path/to/MiniCPM4.1-8B-Q4_K_M.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 \
    --conversation
```

### 混合思考

MiniCPM 4.1 的 chat template 支持 `enable_thinking`。开启思考请同时传 `--jinja` 和 `--chat-template-kwargs`：

```bash
./build/bin/llama-cli \
    -m /path/to/MiniCPM4.1-8B-Q4_K_M.gguf \
    -c 8192 --temp 0.6 --top-p 0.95 \
    --jinja --chat-template-kwargs '{"enable_thinking": true}' \
    -p "列车 9:00 出发，以 80 km/h 行驶到 200 km 外的城市，几点到达？"
```

> [!TIP]
> `enable_thinking` 默认为 `false`。推荐采样：开启时 `temperature=0.6, top_p=0.95`；关闭时 `0.7, 0.8`。

## 4. 从 PyTorch 转换（可选）

```bash
python ./convert_hf_to_gguf.py /path/to/MiniCPM4.1-8B \
    --outfile /path/to/MiniCPM4.1-8B-F16.gguf --outtype f16

./build/bin/llama-quantize \
    /path/to/MiniCPM4.1-8B-F16.gguf \
    /path/to/MiniCPM4.1-8B-Q4_K_M.gguf \
    Q4_K_M
```

## 5. 参数说明

| 参数 | 说明 |
| :--- | :--- |
| `-m, --model` | GGUF 路径 |
| `-p, --prompt` | 单次 prompt |
| `-c, --ctx-size` | 最大上下文（开启 InfLLM-V2 的构建版本下可达 128K）|
| `--conversation` | 多轮交互模式 |
| `--jinja` | 使用模型自带的 Jinja chat template |
| `--chat-template-kwargs` | 传给 chat template 的 JSON 参数 —— 用于切换 `enable_thinking` |
