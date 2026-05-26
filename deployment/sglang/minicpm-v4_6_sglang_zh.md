# MiniCPM-V 4.6 - SGLang

> [!NOTE]
> MiniCPM-V 4.6 自 [PR #24998](https://github.com/sgl-project/sglang/pull/24998)（2026-05-12 merged）起在官方 SGLang `main` 分支上完整可用。**无需使用 fork**，直接从官方仓库安装即可，确保检出的 commit ≥ 该 PR 的 merge 时间。
>
> MiniCPM-V 4.6 在 `transformers>=5.7.0` 中以独立架构 `MiniCPMV4_6ForConditionalGeneration` 注册，SGLang 的适配也基于这一布局。

MiniCPM-V 4.6 提供两个 checkpoint：

- **Instruct** — [`openbmb/MiniCPM-V-4.6`](https://huggingface.co/openbmb/MiniCPM-V-4.6)
- **Thinking（思考）** — [`openbmb/MiniCPM-V-4.6-Thinking`](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking)

## 1. SGLang 安装

### 从官方 main 源码安装

包含 #24998 的正式 release 出来前，推荐从官方 `main` 安装（请关注 [SGLang Releases](https://github.com/sgl-project/sglang/releases)；正式版发布后可直接 `pip install -U "sglang[all]"`）：

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[all]"
```

上述命令会自动安装 `transformers>=5.7.0`，而它会牵入一个较新版本的
PyTorch（截至撰写时 ≥ 2.6）。**安装 FlashInfer 之前**先确认实际 torch /
CUDA 版本，避免下一步装错 wheel 把 torch 又降回去：

```bash
python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__)"
```

### FlashInfer 依赖（可选，建议安装）

> [!IMPORTANT]
> FlashInfer 的 wheel 是按 `(torch 版本, CUDA 版本)` 组合发布的，**必须**
> 选与刚才验证出来的 torch + CUDA 完全匹配的 index。**不要直接复制
> `cu121/torch2.4` 这种老链接**——那会偷偷把 torch 降级并破坏前面装好
> 的 SGLang / transformers 环境。

FlashInfer 总索引：<https://flashinfer.ai/whl/>。挑跟你环境匹配的目录，例如：

| 你的 torch / CUDA           | Index URL                                            |
| :--------------------------- | :--------------------------------------------------- |
| torch 2.6 + CUDA 12.4        | <https://flashinfer.ai/whl/cu124/torch2.6/>          |
| torch 2.6 + CUDA 12.6        | <https://flashinfer.ai/whl/cu126/torch2.6/>          |
| torch 2.7 + CUDA 12.8        | <https://flashinfer.ai/whl/cu128/torch2.7/>          |

然后任选一种安装：

```bash
# 方法 1：从对应索引 pip 安装（国内访问可能慢）
pip install flashinfer-python -i <上表选的 index URL>

# 方法 2：手动下载匹配的 whl 安装
#   1) 浏览器打开索引 URL，选一份匹配你 Python 版本（cp310/cp311/…）
#      与平台（linux_x86_64 / win_amd64）的 whl
#   2) pip install <下载的 whl 文件>
```

其它（Docker 镜像、纯 CPU 退路等）参见
[SGLang 官方安装文档](https://docs.sglang.ai/start/install.html)。

## 2. 启动推理服务

默认情况下，会从 Hugging Face Hub 下载模型文件：

```bash
python -m sglang.launch_server --model-path openbmb/MiniCPM-V-4.6 --port 30000 --trust-remote-code --dtype bfloat16
```

也可以在 `--model-path` 后指定本地路径：

```bash
python -m sglang.launch_server --model-path /your/local/MiniCPM-V-4.6 --port 30000 --trust-remote-code --dtype bfloat16
```

部署 Thinking 模型只需替换 model id：

```bash
python -m sglang.launch_server --model-path openbmb/MiniCPM-V-4.6-Thinking --port 30000 --trust-remote-code --dtype bfloat16
```

## 3. 调用服务接口

- bash / curl：

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

- Python（OpenAI 客户端）调用：

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:30000/v1", api_key="None")

    response = client.chat.completions.create(
        model="MiniCPM-V-4.6",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片"},
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

> 如 `image_url` 无法访问，可替换为本地图片路径或 base64 data URL。
>
> v4.6 使用 Qwen3.5 词表 —— 若发现模型在回答后继续生成，请显式传入 `stop_token_ids=[248044, 248046]`。
>
> 更多调用方法可参考 [SGLang 使用文档](https://docs.sglang.ai/backend/openai_api_vision.html)。
