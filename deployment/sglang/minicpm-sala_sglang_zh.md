# MiniCPM-SALA - SGLang

> [!NOTE]
> SALA 支持位于 [`tc-mb/sglang`](https://github.com/tc-mb/sglang) fork 的 `minicpm_sala` 分支（独立于 MiniCPM 4 / 4.1 使用的 `minicpm` 分支）。该分支包含 SALA 所需的稀疏 + 线性混合注意力 kernel。

## 1. 环境准备

```bash
git clone https://github.com/tc-mb/sglang.git
cd sglang
git checkout minicpm_sala

pip install --upgrade pip
pip install -e "python[all]"
```

> [!IMPORTANT]
> SALA 的稀疏注意力层依赖 [InfLLM-V2 CUDA kernel](https://github.com/OpenBMB/infllmv2_cuda_impl)。启动服务前先 build 安装：
>
> ```bash
> git clone https://github.com/OpenBMB/infllmv2_cuda_impl.git
> cd infllmv2_cuda_impl
> pip install -e .
> ```

## 2. 启动服务

```bash
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM-SALA \
    --port 30000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --context-length 131072
```

> [!TIP]
> SALA 的核心卖点是超长上下文。`--context-length` 按实际需求设置即可——线性注意力层把内存控制住，但稀疏层的 KV 缓存仍会随长度增长。

## 3. 调用服务

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")

with open("long_document.txt", "r", encoding="utf-8") as f:
    long_text = f.read()

resp = client.chat.completions.create(
    model="openbmb/MiniCPM-SALA",
    messages=[{
        "role": "user",
        "content": f"以下是一篇长文档，请阅读后回答末尾的问题。\n\n{long_text}\n\n问题：第三节的主要观点是什么？"
    }],
    temperature=0.7, top_p=0.8, max_tokens=1024,
)
print(resp.choices[0].message.content)
```

## 4. 注意事项

- 使用分支：[`tc-mb/sglang @ minicpm_sala`](https://github.com/tc-mb/sglang/tree/minicpm_sala)。
- Kernel 依赖：[`OpenBMB/infllmv2_cuda_impl`](https://github.com/OpenBMB/infllmv2_cuda_impl)。稀疏注意力层必须依赖它，否则模型会回落到很慢的参考实现。
- SALA 是研究版本，API 与行为可能演进。
