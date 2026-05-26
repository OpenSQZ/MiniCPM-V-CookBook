# MiniCPM 5 - MLX（Apple Silicon）

> [!NOTE]
> [`openbmb/MiniCPM5-1B-MLX`](https://huggingface.co/openbmb/MiniCPM5-1B-MLX) 是基于 [`mlx-lm`](https://github.com/ml-explore/mlx-lm) 的官方 Apple Silicon 优化版本。对 MiniCPM 5 而言，它是 Apple Silicon（M1–M4）上**最高吞吐 + 单 Python 进程**的推荐路径 —— 无需启独立 server、不需要构建 `llama.cpp`。

## 1. 环境准备

```bash
pip install "mlx-lm>=0.31"
```

已在 macOS 14+ / Python 3.10+ 上验证。

## 2. CLI 生成

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

> 💡 `--extra-eos-token "<|im_end|>"`（CLI）或把 `<|im_end|>` 加进包装器的 stop 列表（Python）是**必需**的：发布的 metadata 只把 `</s>` 注册为 EOS，而 chat template 实际是用 `<|im_end|>` **字符串**结束每一轮。缺这个 EOS 会让模型继续生成到下一轮 prompt。

## 3. Python API（流式）

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
    sampler=None,                    # 默认 temp / top_p
):
    print(resp.text, end="", flush=True)
print()
```

也可以直接用 `apply_chat_template` —— MLX 仓库的 `chat_template.jinja` 与 HF 发布版完全一致：

```python
prompt = tk.apply_chat_template(
    [{"role": "user", "content": "用一句话解释什么是 GQA。"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,        # 自动注入 <think>\n
)
```

## 4. 从自有权重生成 MLX（进阶）

如果你训练了自己的 HF fp16 权重并想生成 MLX 版本，用 `mlx_lm.convert`：

```bash
HF=/path/to/your-fp16-hf

# bf16 主副本
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-bf16

# 4-bit（更小 / 更快，平均 ~4.5 bits/weight）
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-q4 -q --q-bits 4
```

4-bit 转换会打印 `[INFO] Quantized model with 4.501 bits per weight.` —— 略高于 4 是因为 `embed_tokens` 和 `lm_head` 保留了较高精度，对小词表 head 模型有利于保质。

## 5. 启动 server

`mlx-lm` 自带 OpenAI 兼容 server：

```bash
mlx_lm.server --model openbmb/MiniCPM5-1B-MLX --port 8000
```

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
print(client.chat.completions.create(
    model="openbmb/MiniCPM5-1B-MLX",
    messages=[{"role": "user", "content": "你好。"}],
).choices[0].message.content)
```

## 6. 推荐采样参数

| 模式 | `--temp` | `--top-p` | 适用场景 |
| :--- | :---: | :---: | :--- |
| Think | 0.9 | 0.95 | 推理、数学、代码、多步任务（模型自动输出 `<think>` 块） |
| No-Think | 0.7 | 0.95 | 快速对话、延迟敏感场景 |

两种模式仅靠采样参数即可激活 —— 发布的 chat template 在没有 `system` 消息屏蔽时会自动注入 `<think>\n`，因此默认就是 Think 行为。

## 7. 常见问题

### 模型一直不停生成

添加 `--extra-eos-token "<|im_end|>"`（CLI）或把 `<|im_end|>` 写进 `stop`（Python）。详见上面的提示。

## 相关文档

- [`transformers`](../inference/chat.html) —— 同一份权重的 CPU / CUDA 路径
- [`llama.cpp`](llamacpp.html) —— 备选的端侧路径（CPU + Metal）
