# MiniCPM-V 4.7 — GGUF Quantization Guide

This guide walks through converting the MiniCPM-V 4.7 PyTorch model to GGUF and quantizing it.
The resulting GGUFs are used with [`llama.cpp`](../../deployment/llama.cpp/minicpm-v4_7_llamacpp.md) or
[`ollama`](../../deployment/ollama/minicpm-v4_7_ollama.md).

> [!IMPORTANT]
> 4.7 uses **canvas M-RoPE**, which introduced a new GGUF contract: the metadata key
> `rope.section_order` and the mmproj tensor `v.tok_embd_sep`. Re-convert with a llama.cpp
> checkout at/after [PR #29416](https://github.com/ggml-org/llama.cpp/pull/29416). GGUFs produced
> by an older converter will compute M-RoPE incorrectly in recent builds.

Pick the size you want to quantize:

- **1B** — <https://huggingface.co/openbmb/MiniCPM-V-4.7-1B>
- **35B-A3B** — <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B>

(Or use the prebuilt GGUFs from `openbmb/MiniCPM-V-4.7-1B-gguf` / `openbmb/MiniCPM-V-4.7-35B-A3B-gguf`.)

### 1. Convert to GGUF

Run from the root of a **current** `llama.cpp` checkout (at/after PR #29416):

```bash
# Step 1 — convert the language model + vision merger to F16 GGUF
python ./convert_hf_to_gguf.py /path/to/MiniCPM-V-4.7-1B \
    --outfile /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-F16.gguf \
    --outtype f16

# Step 2 — convert the vision projector (mmproj)
python ./convert_hf_to_gguf.py /path/to/MiniCPM-V-4.7-1B \
    --mmproj \
    --outfile /path/to/MiniCPM-V-4.7-1B/mmproj-model-f16.gguf
```

`convert_hf_to_gguf.py` autodetects `MiniCPMV4_7ForConditionalGeneration` from `config.json`.
For the 4× vision variant, copy the model dir, set `"downsample_mode": "4x"` in its
`preprocessor_config.json`, and run the `--mmproj` step again with `mmproj-model-f16-4x.gguf`.

### 2. Quantize to INT4

Once you have the F16 LM GGUF, quantize it with `llama-quantize`:

```bash
./llama-quantize \
    /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-F16.gguf \
    /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-Q4_K_M.gguf \
    Q4_K_M
```

Other common quant types you can swap in: `Q5_K_M`, `Q6_K`, `Q8_0`. Pick based on your
accuracy vs. memory tradeoff. The mmproj file is already small — keep it at `F16` and pair it
with whichever quantization you chose for the LM.

### 3. Mixed quantization for 35B-A3B (recommended)

The 35B-A3B MoE expert tensors account for ~92% of the weights. Lowering just the expert
gate/up/down projections to `Q3_K` (on top of the `Q4_K_M` default mixture), with an imatrix,
gives a ~14.4 GiB model that matches `Q4_K_M` on benchmarks. Put this in `tensor-types.txt`:

```text
ffn_(gate|up|down)_exps=Q3_K
```

Then quantize with the imatrix:

```bash
./llama-quantize \
    --imatrix /path/to/v47-35a3.imatrix.gguf \
    --tensor-type-file ./tensor-types.txt \
    /path/to/MiniCPM-V-4.7-35B-A3B/MiniCPM-V-4_7-35B-A3B-F16.gguf \
    /path/to/MiniCPM-V-4.7-35B-A3B/MiniCPM-V-4_7-35B-A3B-mix_q3gud_im.gguf \
    Q4_K_M 32
```

> `--tensor-type-file` rules are **first-match-wins** — put more specific rules first.

## Recommended sampling parameters

For instruct-mode generation we recommend: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.
