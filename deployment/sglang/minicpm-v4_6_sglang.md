# MiniCPM-V 4.6 - SGLang Documentation

> [!NOTE]
> MiniCPM-V 4.6 is supported on the official SGLang `main` branch since [PR #24998](https://github.com/sgl-project/sglang/pull/24998) (merged 2026-05-12). **No fork is required** — install directly from upstream, making sure your checkout is at or after that commit.
>
> MiniCPM-V 4.6 is registered in `transformers>=5.7.0` as a standalone architecture (`MiniCPMV4_6ForConditionalGeneration`); the SGLang adapter follows that layout.

MiniCPM-V 4.6 ships as two checkpoints:

- **Instruct** — [`openbmb/MiniCPM-V-4.6`](https://huggingface.co/openbmb/MiniCPM-V-4.6)
- **Thinking** — [`openbmb/MiniCPM-V-4.6-Thinking`](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking)

## 1. Installing SGLang

### Install SGLang from upstream main

Until a tagged SGLang release ships with #24998, install from upstream `main` (watch [SGLang Releases](https://github.com/sgl-project/sglang/releases); once a release includes that PR, `pip install -U "sglang[all]"` is enough):

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[all]"
```

`transformers>=5.7.0` is installed automatically — this in turn requires a
recent PyTorch (≥ 2.6 at the time of writing). Verify the resolved versions
match what FlashInfer needs *before* installing FlashInfer below:

```bash
python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__)"
```

### Installing FlashInfer (optional but recommended)

> [!IMPORTANT]
> FlashInfer wheels are pinned to a specific `(torch, cuda)` combo. Pick the
> wheel index that matches the **torch + CUDA you just verified above** —
> don't blindly copy a `cu121/torch2.4` URL, that will silently downgrade
> torch and break the SGLang / transformers install.

The general index lives at <https://flashinfer.ai/whl/>. Pick the directory
matching your environment, for example:

| Your torch / CUDA            | Index URL                                            |
| :--------------------------- | :--------------------------------------------------- |
| torch 2.6 + CUDA 12.4        | <https://flashinfer.ai/whl/cu124/torch2.6/>          |
| torch 2.6 + CUDA 12.6        | <https://flashinfer.ai/whl/cu126/torch2.6/>          |
| torch 2.7 + CUDA 12.8        | <https://flashinfer.ai/whl/cu128/torch2.7/>          |

Then install via either:

```bash
# Method 1 — pip from the right index (slow / blocked in CN)
pip install flashinfer-python -i <index URL from table above>

# Method 2 — download the matching wheel manually
#   1) Open the index URL in a browser, find a wheel that matches your
#      python version (cp310 / cp311 / ...) and platform (linux_x86_64 / win_amd64)
#   2) pip install <downloaded-wheel.whl>
```

For everything else (Docker images, CPU-only fallback, etc.) see the
[official SGLang installation docs](https://docs.sglang.ai/start/install.html).

## 2. Launching the Inference Server

By default the server downloads weights from the HuggingFace Hub:

```bash
python -m sglang.launch_server --model-path openbmb/MiniCPM-V-4.6 --port 30000 --trust-remote-code --dtype bfloat16
```

Or specify a local path:

```bash
python -m sglang.launch_server --model-path /your/local/MiniCPM-V-4.6 --port 30000 --trust-remote-code --dtype bfloat16
```

To serve the Thinking variant, swap the model id:

```bash
python -m sglang.launch_server --model-path openbmb/MiniCPM-V-4.6-Thinking --port 30000 --trust-remote-code --dtype bfloat16
```

## 3. Calling the Service

- Bash / curl:

    ```bash
    curl -s http://localhost:30000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "MiniCPM-V-4.6",
        "messages": [
          {
            "role": "user",
            "content": [
              {"type": "text", "text": "What is in this image?"},
              {
                "type": "image_url",
                "image_url": {
                  "url": "https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/inference/assets/airplane.jpeg?raw=true"
                }
              }
            ]
          }
        ],
        "max_tokens": 300,
        "top_k": -1
      }'
    ```

- Python (OpenAI client):

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:30000/v1", api_key="None")

    response = client.chat.completions.create(
        model="MiniCPM-V-4.6",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/inference/assets/airplane.jpeg?raw=true",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
        extra_body={"top_k": -1, "stop_token_ids": [248044, 248046]},
    )

    print(response.choices[0].message.content)
    ```

> If `image_url` is not reachable from your machine, replace it with a local path / base64 data URL.
>
> v4.6 uses the Qwen3.5 vocabulary — pass `stop_token_ids = [248044, 248046]` if you observe the model continuing past the answer.
>
> For more invocation patterns, see the [SGLang documentation](https://docs.sglang.ai/backend/openai_api_vision.html).
