# MiniCPM 5 - MLX (Apple Silicon)

> [!NOTE]
> [`openbmb/MiniCPM5-1B-MLX`](https://huggingface.co/openbmb/MiniCPM5-1B-MLX) is the official Apple-Silicon-optimised release using [`mlx-lm`](https://github.com/ml-explore/mlx-lm). For MiniCPM 5 it is the recommended path on Apple Silicon (M1–M4) when you want **highest throughput** and want to stay inside one Python process — no separate server, no `llama.cpp` build chain.

## 1. Environment

```bash
pip install "mlx-lm>=0.31"
```

Tested on macOS 14+ with Python 3.10+.

## 2. Generate from the CLI

```bash
mlx_lm.generate \
    --model openbmb/MiniCPM5-1B-MLX \
    --prompt "<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
" \
    --max-tokens 200 --temp 0.7 --top-p 0.95 \
    --extra-eos-token "<|im_end|>"
```

> 💡 `--extra-eos-token "<|im_end|>"` (CLI) or adding `<|im_end|>` to the wrapper's stop list (Python) is required: the released metadata only registers `</s>` as model EOS, but the chat template ends turns with the `<|im_end|>` *string*. Without an extra EOS the model will keep generating into the next role's prompt.

## 3. Python API (streaming)

```python
from mlx_lm import load, stream_generate

model, tk = load("openbmb/MiniCPM5-1B-MLX")

prompt = (
    "<|im_start|>user\n"
    "用一句话解释什么是 GQA。<|im_end|>\n"
    "<|im_start|>assistant\n"
)

for resp in stream_generate(
    model, tk, prompt=prompt, max_tokens=512,
    sampler=None,                    # default temp / top_p
):
    print(resp.text, end="", flush=True)
print()
```

You can also use `apply_chat_template` directly — the MLX repo carries the same `chat_template.jinja` as the HF release:

```python
prompt = tk.apply_chat_template(
    [{"role": "user", "content": "用一句话解释什么是 GQA。"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,        # auto-emits <think>\n
)
```

## 4. Building MLX weights from your own checkpoint (advanced)

If you have a self-trained HF fp16 checkpoint and want to produce MLX weights, use `mlx_lm.convert`:

```bash
HF=/path/to/your-fp16-hf

# bf16 master copy
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-bf16

# 4-bit (smaller / faster, ~4.5 bits/weight on average)
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-q4 -q --q-bits 4
```

The 4-bit pass logs `[INFO] Quantized model with 4.501 bits per weight.` — the slight overshoot above 4 bits is from keeping `embed_tokens` and `lm_head` in higher precision, which preserves quality on the small, untied vocabulary head.

## 5. Run as a server

`mlx-lm` ships an OpenAI-compatible server:

```bash
mlx_lm.server --model openbmb/MiniCPM5-1B-MLX --port 8000
```

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
print(client.chat.completions.create(
    model="openbmb/MiniCPM5-1B-MLX",
    messages=[{"role": "user", "content": "Hello."}],
).choices[0].message.content)
```

## 6. Recommended sampling

| Mode | `--temp` | `--top-p` | When to use |
| :--- | :---: | :---: | :--- |
| Think | 0.9 | 0.95 | reasoning, math, code, multi-step (model auto-emits `<think>` block) |
| No-Think | 0.7 | 0.95 | fast assistant, latency-bound |

Both modes are activated by sampling parameters only — the released chat template auto-injects `<think>\n` when no `system` message disables it, so you get Think-mode behaviour by default.

## 7. Q&A

### Model never stops generating

Add `--extra-eos-token "<|im_end|>"` (CLI) or include `<|im_end|>` in `stop` (Python). See the note above.

## See also

- [`transformers`](../inference/chat.html) — same checkpoint, CPU / CUDA path
- [`llama.cpp`](llamacpp.html) — alternative on-device path (CPU + Metal)
