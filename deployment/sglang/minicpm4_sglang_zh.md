# MiniCPM 4 - SGLang

> [!NOTE]
> SGLang 上游尚未合入 MiniCPM 4 支持，因此我们在 [`tc-mb/sglang`](https://github.com/tc-mb/sglang) fork 的 `minicpm` 分支上维护。与 MiniCPM 4.1 共用同一分支 —— pull 一次即可在 4 与 4.1 之间自由切换。（SALA 位于独立的 [`minicpm_sala`](https://github.com/tc-mb/sglang/tree/minicpm_sala) 分支。）

## 1. 环境准备

```bash
git clone https://github.com/tc-mb/sglang.git
cd sglang
git checkout minicpm

pip install --upgrade pip
pip install -e "python[all]"
```

## 2. 启动服务

```bash
# 8B
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM4-8B \
    --port 30000 --trust-remote-code --dtype bfloat16 --context-length 32768

# 0.5B（端侧友好）
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM4-0.5B \
    --port 30000 --trust-remote-code --dtype bfloat16 --context-length 32768
```

OpenAI 兼容接口在 `http://localhost:30000/v1`。

## 3. 调用服务

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")

resp = client.chat.completions.create(
    model="openbmb/MiniCPM4-8B",
    messages=[{"role": "user", "content": "写一篇关于人工智能的文章。"}],
    temperature=0.7, top_p=0.8, max_tokens=512,
)
print(resp.choices[0].message.content)
```

## 4. 注意事项

- 使用分支：[`tc-mb/sglang @ minicpm`](https://github.com/tc-mb/sglang/tree/minicpm)。
- MiniCPM 4 不支持 `enable_thinking` 开关。需要混合思考请使用 [MiniCPM 4.1](../minicpm4_1/deployment/sglang.html)。
- 超长输入参考 [MiniCPM-SALA](../minicpm-sala/deployment/sglang.html)。
