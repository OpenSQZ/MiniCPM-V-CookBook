# MiniCPM-V 4.7 — llama.cpp

> [!NOTE]
> MiniCPM-V 4.7 support has been merged into the official `llama.cpp`
> ([PR #29416](https://github.com/ggml-org/llama.cpp/pull/29416)). Build from a checkout at or
> after that PR.
>
> **GGUF format change:** 4.7 uses canvas M-RoPE, which introduced a new GGUF metadata key
> (`rope.section_order`) and a new mmproj tensor (`v.tok_embd_sep`). GGUF files converted with an
> **older** `convert_hf_to_gguf.py` will load in recent builds but compute M-RoPE incorrectly (or
> fail on the missing mmproj tensor). Always use the [official 4.7 GGUFs](#2-gguf-files) or
> re-convert with the current script.

MiniCPM-V 4.7 comes in two sizes; use the matching GGUF set:

| Size | LLM file | mmproj |
|---|---|---|
| 1B | `MiniCPM-V-4_7-1B-Q4_K_M.gguf` | `mmproj-model-f16.gguf` |
| 35B-A3B | `MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf` | `mmproj-model-f16.gguf` |

## 1. Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

Build with CMake ([build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)):

**CPU / Metal:**

```bash
cmake -B build
cmake --build build --config Release
```

**CUDA:**

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

## 2. GGUF files

### Option 1: Download official GGUF files

Download the language-model GGUF and the vision projector (`mmproj`) from:

- HuggingFace:
  - 1B — <https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf>
  - 35B-A3B — <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf>
- ModelScope — `OpenBMB/MiniCPM-V-4.7-1B-gguf` / `OpenBMB/MiniCPM-V-4.7-35B-A3B-gguf`

Each repository ships the full K-quant set (`F16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q5_K_S`, `Q5_1`, `Q5_0`, `Q4_K_M`, `Q4_K_S`, `Q4_1`, `Q4_0`) plus the two mmproj files (`mmproj-model-f16.gguf`, `mmproj-model-f16-4x.gguf`). The 35B-A3B repository additionally ships `mix_q3gud_im` (a 14.4 GiB imatrix mixture recommended for memory-constrained devices).

### Option 2: Convert from PyTorch model

Download the PyTorch checkpoint (e.g. `openbmb/MiniCPM-V-4.7-1B`) and run the standard script from a **current** llama.cpp checkout:

```bash
# 1) Convert the language model + vision merger to GGUF
python ./convert_hf_to_gguf.py /path/to/MiniCPM-V-4.7-1B \
    --outfile /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-F16.gguf \
    --outtype f16

# 2) Convert the vision projector (mmproj)
python ./convert_hf_to_gguf.py /path/to/MiniCPM-V-4.7-1B \
    --mmproj \
    --outfile /path/to/MiniCPM-V-4.7-1B/mmproj-model-f16.gguf
```

To export the 4× vision variant, copy the model directory, set `"downsample_mode": "4x"` in its `preprocessor_config.json`, and run the `--mmproj` step again with `--outfile mmproj-model-f16-4x.gguf`.

## 3. Model Inference

> [!IMPORTANT]
> `llama.cpp` always injects `enable_thinking` into the chat template (its own default is `true`),
> so the template's default is **not** what decides this — the runtime is. The GGUFs at the root of
> the repository pre-disable thinking inside the embedded template, so **plain assistant replies are
> the default and no flag is needed**. Passing `--reasoning on` has no effect on those files.
> For a thinking-enabled build, use the `thinking/` subfolder — see
> [Thinking mode](#thinking-mode).

```bash
cd build/bin/

# F16 weights (1B)
./llama-mtmd-cli \
    -m  /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-F16.gguf \
    --mmproj /path/to/MiniCPM-V-4.7-1B/mmproj-model-f16.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 \
    --reasoning off \
    --image xx.jpg -p "What is in the image?"

# Quantized INT4 weights (35B-A3B)
./llama-mtmd-cli \
    -m  /path/to/MiniCPM-V-4.7-35B-A3B/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf \
    --mmproj /path/to/MiniCPM-V-4.7-35B-A3B/mmproj-model-f16.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 \
    --reasoning off \
    --image xx.jpg -p "What is in the image?"

# Interactive mode
./llama-mtmd-cli \
    -m  /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-Q4_K_M.gguf \
    --mmproj /path/to/MiniCPM-V-4.7-1B/mmproj-model-f16.gguf \
    -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 \
    --reasoning off \
    --image xx.jpg -i
```

### Thinking mode

The repository root GGUFs cannot think: their embedded template always emits an empty think block,
so the model answers directly regardless of `--reasoning`. Two ways to get a reasoning trace:

1. **Use the `thinking/` variants** (recommended). Every GGUF repository ships the same quantization
   levels under `thinking/` with a `-thinking` suffix; those files restore the upstream template and
   are identical to the root files except for the embedded template:

   ```bash
   llama-cli -m thinking/MiniCPM-V-4_7-1B-Q4_K_M-thinking.gguf \
             --mmproj mmproj-model-f16.gguf --image xx.jpg -p "What is in the image?" -n 1024
   ```

   Thinking is on by default there; pass `--reasoning off` to suppress it.

2. **Use the Transformers weights** with `enable_thinking=True`, or build your own GGUF from the
   unmodified template.

> The model is not trained as a dedicated reasoning model — the trace is useful for inspecting
> intermediate steps, but it also multiplies the generated token count, so raise `-n` accordingly.

For reference, `--jinja` is on by default and the underlying flag is `--reasoning on`:

```bash
./llama-mtmd-cli \
    -m  /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-Q4_K_M.gguf \
    --mmproj /path/to/MiniCPM-V-4.7-1B/mmproj-model-f16.gguf \
    -c 8192 --image xx.jpg --jinja --reasoning on -p "What is in the image?"
```

**Argument Reference:**

| Argument | Description |
| :--- | :--- |
| `-m, --model` | Path to the language model GGUF |
| `--mmproj` | Path to the vision projector GGUF |
| `--image` | Path to the input image |
| `-p, --prompt` | The prompt |
| `-c, --ctx-size` | Maximum context size |
| `-rea, --reasoning [on\|off\|auto]` | Enable / disable thinking. Default `auto`. **These GGUFs hard-disable thinking, so this flag has no effect** |
| `--jinja` | Use the model's Jinja chat template (enabled by default in recent builds) |

## 4. Speculative decoding with DSpark

MiniCPM-V 4.7 ships **DSpark** draft heads that accelerate generation with speculative decoding. Download the draft GGUF that matches your target size:

- HuggingFace: `openbmb/MiniCPM-V-4.7-1B-DSpark-gguf` / `openbmb/MiniCPM-V-4.7-35B-A3B-DSpark-gguf`
- ModelScope: `OpenBMB/MiniCPM-V-4.7-1B-DSpark-gguf` / `OpenBMB/MiniCPM-V-4.7-35B-A3B-DSpark-gguf`

Each repository ships the same K-quant set as the target (`F16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q5_K_S`, `Q5_1`, `Q5_0`, `Q4_K_M`, `Q4_K_S`, `Q4_1`, `Q4_0`). Speculative decoding is exact — the target verifies every draft token — so draft quantization only affects acceptance length (speed), never the output; pick the smallest quant that fits, `Q8_0` / `Q6_K` are good defaults.

> [!IMPORTANT]
> `llama-mtmd-cli` does **not** support draft models. Use `llama-cli` (handles both the image projector and the draft) or `llama-server`.

```bash
cd build/bin/

# Image + speculative decoding
./llama-cli \
    -m  /path/to/MiniCPM-V-4.7-1B/MiniCPM-V-4_7-1B-Q4_K_M.gguf \
    --mmproj /path/to/MiniCPM-V-4.7-1B/mmproj-model-f16.gguf \
    -md /path/to/MiniCPM-V-4.7-1B-DSpark-gguf/dspark-MiniCPM-V-4_7-1B-Q8_0.gguf \
    --spec-type draft-dspark \
    -c 8192 --reasoning off \
    --image xx.jpg -p "What is in the image?"

# Serve instead of using the CLI
./llama-server \
    -m  /path/to/MiniCPM-V-4.7-35B-A3B/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf \
    --mmproj /path/to/MiniCPM-V-4.7-35B-A3B/mmproj-model-f16.gguf \
    -md /path/to/MiniCPM-V-4.7-35B-A3B-DSpark-gguf/dspark-MiniCPM-V-4_7-35B-A3B-Q8_0.gguf \
    --spec-type draft-dspark --spec-draft-n-max 15 \
    -c 8192 --jinja --port 8080
```

The draft is paired with one specific target; always use the draft whose name matches the target size. `llama-server` reports the accepted length in each response (`draft_n` / `draft_n_accepted`).

## Recommended sampling parameters

For instruct-mode generation we recommend: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.
