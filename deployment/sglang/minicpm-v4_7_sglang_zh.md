# MiniCPM-V 4.7 — SGLang 文档

> [!NOTE]
> MiniCPM-V 4.7 **暂未合入上游 SGLang**。请使用 fork
> [`tc-mb/sglang`](https://github.com/tc-mb/sglang) 的 `Support-MiniCPM-V4.7` 分支。
> 上游 PR 进行中，合入后可直接安装 SGLang `main`。

MiniCPM-V 4.7 提供两个尺寸，改 `--model-path` 即可切换：

- **1B** — [`openbmb/MiniCPM-V-4.7-1B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)
- **35B-A3B** — [`openbmb/MiniCPM-V-4.7-35B-A3B`](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)

## 1. 安装 SGLang

```bash
git clone -b Support-MiniCPM-V4.7 https://github.com/tc-mb/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[all]"
```

需要 `transformers >= 5.18.0`（4.7 在其中为原生架构）。验证：

```bash
python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__)"
```

### FlashInfer 依赖

fork [将 FlashInfer 声明为依赖](https://github.com/tc-mb/sglang/blob/Support-MiniCPM-V4.7/python/pyproject.toml)，
因此上面的可编辑安装在正常解析依赖时会包含它；这并不证明在你的环境中安装能够成功。
不要再从无关的 torch/CUDA wheel 索引单独安装 FlashInfer，
以免覆盖当前 fork 所需的版本。创建或更新环境时，应保持 fork 声明的
Torch、CUDA 和 FlashInfer 版本组合，并检查所用修订版本中的依赖文件。

通用平台前提见 [SGLang 官方安装文档](https://docs.sglang.io/docs/get-started/install)。
上游标准安装包或 Docker 镜像不能替代上文要求的 MiniCPM-V 4.7 专用 fork。

## 2. 启动推理服务

```bash
# 1B
python -m sglang.launch_server --model-path openbmb/MiniCPM-V-4.7-1B --port 30000 --dtype bfloat16

# 35B-A3B（需大显存节点；用 --tp-size 跨卡分片）
python -m sglang.launch_server --model-path openbmb/MiniCPM-V-4.7-35B-A3B --port 30000 --dtype bfloat16 --tp-size 2
```

或指定本地路径：

```bash
python -m sglang.launch_server --model-path /your/local/MiniCPM-V-4.7-1B --port 30000 --dtype bfloat16
```

> 4.7 **不需要** `--trust-remote-code`（在 `transformers >= 5.18.0` 中原生支持）。

## 3. 调用服务

- Bash / curl：

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

- Python（OpenAI 客户端）：

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

> 若 `image_url` 本机不可达，换成本地路径 / base64 data URL。
>
> v4.7 使用 Qwen3.5 词表 —— 若模型在答案后继续输出，传入 `stop_token_ids = [248044, 248046]`。
>
> 如需关闭 `<think>` 块，传 `chat_template_kwargs={"enable_thinking": false}`（或服务端等价默认值）。

## 推荐采样参数

在 instruct 模式下生成时，推荐使用：`temperature=0.7`、`top_p=0.95`、`top_k=100`、`min_p=0.0`、`presence_penalty=0`、`repetition_penalty=1.03`。
