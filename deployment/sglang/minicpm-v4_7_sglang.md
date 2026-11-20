# MiniCPM-V 4.7 — SGLang Documentation

> [!NOTE]
> MiniCPM-V 4.7 support is **not in upstream SGLang yet**. Use the fork
> [`tc-mb/sglang`](https://github.com/tc-mb/sglang) on branch `Support-MiniCPM-V4.7`.
> An upstream PR is in progress; once merged, install from SGLang `main` instead.

MiniCPM-V 4.7 comes in two sizes; swap `--model-path` to pick one:

- **1B** — [`openbmb/MiniCPM-V-4.7-1B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)
- **35B-A3B** — [`openbmb/MiniCPM-V-4.7-35B-A3B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)

## 1. Installing SGLang

```bash
git clone -b Support-MiniCPM-V4.7 https://github.com/tc-mb/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[all]"
```

`transformers >= 5.18.0` is required (4.7 is a native architecture there). Verify:

```bash
python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__)"
```

### FlashInfer dependencies

The fork [declares FlashInfer as a dependency](https://github.com/tc-mb/sglang/blob/Support-MiniCPM-V4.7/python/pyproject.toml),
so normal dependency resolution for the editable install above includes it.
This does not verify that installation succeeds in your environment.
Do not install FlashInfer again from an unrelated torch/CUDA wheel index: doing
so can replace the version required by the checked-out fork. Keep the fork's
Torch, CUDA and FlashInfer dependency versions together when creating or updating
your environment, and check the dependency file at the revision you use.

See the [official SGLang installation docs](https://docs.sglang.io/docs/get-started/install)
for general platform prerequisites. The standard upstream package or Docker image
does not replace the MiniCPM-V 4.7 fork required above.

## 2. Launching the Inference Server

```bash
# 1B
python -m sglang.launch_server --model-path openbmb/MiniCPM-V-4.7-1B --port 30000 --dtype bfloat16

# 35B-A3B (needs a large-memory GPU node; add --tp-size to shard across GPUs)
python -m sglang.launch_server --model-path openbmb/MiniCPM-V-4.7-35B-A3B --port 30000 --dtype bfloat16 --tp-size 2
```

Or a local path:

```bash
python -m sglang.launch_server --model-path /your/local/MiniCPM-V-4.7-1B --port 30000 --dtype bfloat16
```

> `--trust-remote-code` is **not** required for 4.7 (native in `transformers >= 5.18.0`).

## 3. Calling the Service

- Bash / curl:

    ```bash
    curl -s http://localhost:30000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "MiniCPM-V-4.7-1B",
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
        model="MiniCPM-V-4.7-1B",
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
> v4.7 uses the Qwen3.5 vocabulary — pass `stop_token_ids = [248044, 248046]` if the model continues past the answer.
>
> To disable the `<think>` block, pass `chat_template_kwargs={"enable_thinking": false}` (or the equivalent server default).

## Recommended sampling parameters

For instruct-mode generation we recommend: `temperature=0.7`, `top_p=0.95`, `top_k=100`, `min_p=0.0`, `presence_penalty=0`, `repetition_penalty=1.03`.
