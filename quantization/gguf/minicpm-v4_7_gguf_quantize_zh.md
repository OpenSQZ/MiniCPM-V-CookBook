# MiniCPM-V 4.7 — GGUF 量化指南

本指南演示如何把 MiniCPM-V 4.7 的 PyTorch 模型转换为 GGUF 并量化。
产出的 GGUF 用于 [`llama.cpp`](../../deployment/llama.cpp/minicpm-v4_7_llamacpp.md) 或
[`ollama`](../../deployment/ollama/minicpm-v4_7_ollama.md)。

> [!IMPORTANT]
> 4.7 使用 **canvas M-RoPE**，引入了新的 GGUF 契约：元数据键 `rope.section_order` 与 mmproj 张量
> `v.tok_embd_sep`。请使用 [PR #29416](https://github.com/ggml-org/llama.cpp/pull/29416) 之后（含）
> 的 llama.cpp 重新转换；旧版转换器产出的 GGUF 在近期构建里 M-RoPE 会算错。

选择要量化的尺寸：

- **1B** — <https://huggingface.co/openbmb/MiniCPM-V-4.7-1B>
- **35B-A3B** — <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B>

（或直接使用预构建 GGUF：`openbmb/MiniCPM-V-4.7-1B-gguf` / `openbmb/MiniCPM-V-4.7-35B-A3B-gguf`。）

### 1. 转换为 GGUF

在**当前版本**（含 PR #29416）的 `llama.cpp` 根目录执行：

```bash
# 步骤 1 — 转换语言模型 + 视觉 merger 为 F16 GGUF
python ./convert_hf_to_gguf.py /path/to/MiniCPM-V-4.7-1B \
    --outfile /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-F16.gguf \
    --outtype f16

# 步骤 2 — 转换视觉投影（mmproj）
python ./convert_hf_to_gguf.py /path/to/MiniCPM-V-4.7-1B \
    --mmproj \
    --outfile /path/to/MiniCPM-V-4.7-1B/mmproj-model-f16.gguf
```

`convert_hf_to_gguf.py` 会从 `config.json` 自动识别 `MiniCPMV4_7ForConditionalGeneration`。
4× 视觉变体：复制模型目录，把其 `preprocessor_config.json` 里的 `"downsample_mode"` 改成 `"4x"`，
再跑一次 `--mmproj` 输出 `mmproj-model-f16-4x.gguf`。

### 2. 量化为 INT4

拿到 F16 LM GGUF 后，用 `llama-quantize` 量化：

```bash
./llama-quantize \
    /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-F16.gguf \
    /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-Q4_K_M.gguf \
    Q4_K_M
```

其他常用档位：`Q5_K_M`、`Q6_K`、`Q8_0`，按精度/显存权衡选择。mmproj 体积已很小，保持 `F16`，
与所选 LM 量化搭配即可。

### 3. 35B-A3B 的混合量化（推荐）

35B-A3B 的 MoE 专家张量约占权重 92%。在 `Q4_K_M` 默认 mixture 之上，只把专家 gate/up/down
降到 `Q3_K`，并配合 imatrix，可得到约 14.4 GiB、精度与 `Q4_K_M` 相当的模型。`tensor-types.txt` 内容：

```text
ffn_(gate|up|down)_exps=Q3_K
```

带 imatrix 量化：

```bash
./llama-quantize \
    --imatrix /path/to/v47-35a3.imatrix.gguf \
    --tensor-type-file ./tensor-types.txt \
    /path/to/MiniCPM-V-4.7-35B-A3B/MiniCPM-V-4_7-35B-A3B-F16.gguf \
    /path/to/MiniCPM-V-4.7-35B-A3B/MiniCPM-V-4_7-35B-A3B-mix_q3gud_im.gguf \
    Q4_K_M 32
```

> `--tensor-type-file` 是**首个匹配生效** —— 更具体的规则要放前面。
