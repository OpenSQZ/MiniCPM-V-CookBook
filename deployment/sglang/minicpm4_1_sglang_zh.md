# MiniCPM 4.1 - SGLang

> [!NOTE]
> SGLang 上游尚未合入 MiniCPM 4.1 支持，因此我们在 [`tc-mb/sglang`](https://github.com/tc-mb/sglang) fork 的 `minicpm` 分支上维护。该分支跟踪 SGLang `main` 并叠加 MiniCPM 4 / 4.1 修改，定期 rebase。API 稳定后会推动合入上游。（SALA 位于独立的 [`minicpm_sala`](https://github.com/tc-mb/sglang/tree/minicpm_sala) 分支。）

## 1. 环境准备

```bash
git clone https://github.com/tc-mb/sglang.git
cd sglang
git checkout minicpm

pip install --upgrade pip
pip install -e "python[all]"
```

> [!TIP]
> SGLang 默认 attention 后端依赖 FlashInfer。安装失败时可先手动安装：
>
> ```bash
> pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/
> ```

## 2. 启动服务

```bash
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM4.1-8B \
    --port 30000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --context-length 65536
```

服务在 `http://localhost:30000/v1` 暴露 OpenAI 兼容接口。

## 3. 调用服务

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")

resp = client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B",
    messages=[{"role": "user", "content": "写一篇关于端侧 AI 的短文。"}],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### 通过 API 开启混合思考

```python
resp = client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B",
    messages=[{"role": "user", "content": "求解：列车 9:00 出发，以 80 km/h 行驶到 200 km 外的城市..."}],
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
print(resp.choices[0].message.content)
```

## 4. 注意事项

- 使用分支：[`tc-mb/sglang @ minicpm`](https://github.com/tc-mb/sglang/tree/minicpm)。生产环境建议 pin 到具体 commit。
- 多卡推理添加 `--tp 2`（tensor parallel size）。MiniCPM 4.1 是单一 8B checkpoint，新一代 GPU 通常不需要 TP。
- 音频 / 图像输入请使用 MiniCPM-V 或 MiniCPM-o。
