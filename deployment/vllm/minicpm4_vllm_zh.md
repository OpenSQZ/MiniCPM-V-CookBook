# MiniCPM 4 - vLLM

> [!NOTE]
> MiniCPM 4 已支持**上游 vLLM**，无需 fork。8B / 0.5B 基础模型、[Eagle-vLLM](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-vLLM) 投机 draft、[marlin-Eagle-vLLM](https://huggingface.co/openbmb/MiniCPM4-8B-marlin-Eagle-vLLM) INT4 + 投机组合包均可直接用官方 `vllm` 包运行。

## 1. 环境准备

```bash
pip install -U vllm
```

> [!TIP]
> 使用 vLLM `>= 0.9.0` 获得一等的 MiniCPM 4 支持。

## 2. 启动服务

```bash
vllm serve openbmb/MiniCPM4-8B \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9
```

超轻量 0.5B 变体：

```bash
vllm serve openbmb/MiniCPM4-0.5B \
    --trust-remote-code \
    --max-model-len 32768
```

服务在 `http://localhost:8000/v1` 暴露 OpenAI 兼容接口。

## 3. 调用服务

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

resp = client.chat.completions.create(
    model="openbmb/MiniCPM4-8B",
    messages=[{"role": "user", "content": "写一篇关于人工智能的文章。"}],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

## 4. 投机解码（Eagle-vLLM）

MiniCPM 4 提供为 vLLM 适配的 EAGLE draft：[`openbmb/MiniCPM4-8B-Eagle-vLLM`](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-vLLM)：

```bash
vllm serve openbmb/MiniCPM4-8B \
    --trust-remote-code \
    --max-model-len 32768 \
    --speculative-config '{"method":"eagle","model":"openbmb/MiniCPM4-8B-Eagle-vLLM","num_speculative_tokens":5}'
```

## 5. Marlin INT4 + 投机

[`openbmb/MiniCPM4-8B-marlin-Eagle-vLLM`](https://huggingface.co/openbmb/MiniCPM4-8B-marlin-Eagle-vLLM) 把 Marlin INT4 权重和配套 Eagle draft 打包在一起，以获得最大吞吐：

```bash
vllm serve openbmb/MiniCPM4-8B-marlin-Eagle-vLLM \
    --trust-remote-code \
    --max-model-len 32768 \
    --speculative-config '{"method":"eagle","model":"openbmb/MiniCPM4-8B-Eagle-vLLM","num_speculative_tokens":5}'
```

Marlin 需要 CUDA 计算能力 ≥ 8.0。

## 6. 离线推理

```python
from vllm import LLM, SamplingParams

llm = LLM(model="openbmb/MiniCPM4-8B", trust_remote_code=True, max_model_len=32768)
sampling = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=512)
outputs = llm.chat(
    [{"role": "user", "content": "写一篇关于人工智能的文章。"}],
    sampling_params=sampling,
)
print(outputs[0].outputs[0].text)
```

## 7. 注意事项

- MiniCPM 4 **不支持** `enable_thinking` 开关。需要混合思考请使用 [MiniCPM 4.1](../minicpm4_1/deployment/vllm.html)。
- `--trust-remote-code` 必须开启。
- vLLM 投机解码请使用 `MiniCPM4-8B-Eagle-vLLM`；`*-Eagle-FRSpec*` 是为另一种运行时调优的，**不**兼容 vLLM。
