# MiniCPM 4.1 - vLLM

> [!NOTE]
> MiniCPM 4.1 已支持**上游 vLLM**，无需 fork 或切换分支。8B 基础模型、[EAGLE3](https://huggingface.co/openbmb/MiniCPM4.1-8B-Eagle3) draft、[Marlin INT4](https://huggingface.co/openbmb/MiniCPM-4.1-8B-Marlin) 变体均可直接用官方 `vllm` 包运行。

## 1. 环境准备

```bash
# 推荐：直接从 PyPI 安装
pip install -U vllm

# 或从源码构建最新上游
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install --editable . -v
```

> [!TIP]
> 使用 vLLM `>= 0.10.0`。更早的版本不包含 MiniCPM 4.1 支持。

## 2. API 服务

### 启动服务

```bash
vllm serve openbmb/MiniCPM4.1-8B \
    --trust-remote-code \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.9
```

默认在 `http://localhost:8000` 暴露 OpenAI 兼容的 `/v1/chat/completions` 接口。

### 调用服务（OpenAI 客户端）

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

resp = client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B",
    messages=[
        {"role": "user", "content": "写一篇关于端侧 AI 的短文。"}
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### 通过 API 启用混合思考

通过 `extra_body.chat_template_kwargs` 传入 `enable_thinking=True` 开启逐步推理：

```python
resp = client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B",
    messages=[
        {"role": "user", "content": "某列车 9:00 出发，以 80 km/h 行驶到 200 km 外的城市，几点到达？"}
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
print(resp.choices[0].message.content)
```

## 3. 离线推理

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="openbmb/MiniCPM4.1-8B",
    trust_remote_code=True,
    max_model_len=65536,
    gpu_memory_utilization=0.9,
)

sampling = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=512)
outputs = llm.chat(
    [{"role": "user", "content": "写一篇关于端侧 AI 的短文。"}],
    sampling_params=sampling,
)
print(outputs[0].outputs[0].text)
```

## 4. EAGLE3 投机解码

MiniCPM 4.1 同步发布了官方 EAGLE3 draft 模型 [`openbmb/MiniCPM4.1-8B-Eagle3`](https://huggingface.co/openbmb/MiniCPM4.1-8B-Eagle3)，在推理任务上可获得最高 **3 倍生成加速**。

```bash
vllm serve openbmb/MiniCPM4.1-8B \
    --trust-remote-code \
    --max-model-len 65536 \
    --speculative-config '{"method":"eagle3","model":"openbmb/MiniCPM4.1-8B-Eagle3","num_speculative_tokens":5}'
```

OpenAI 兼容接口对客户端透明，投机解码无感知。

## 5. Marlin INT4 加速

在 Ampere / Ada / Hopper GPU 上，官方 [`openbmb/MiniCPM-4.1-8B-Marlin`](https://huggingface.co/openbmb/MiniCPM-4.1-8B-Marlin) INT4 权重可进一步降低延迟。只需把启动命令的模型名换掉即可，其他不变：

```bash
vllm serve openbmb/MiniCPM-4.1-8B-Marlin \
    --trust-remote-code \
    --max-model-len 65536
```

Marlin 需要 CUDA 计算能力 ≥ 8.0。更老的 GPU 请改用 AWQ（[`openbmb/MiniCPM4.1-8B-AutoAWQ`](https://huggingface.co/openbmb/MiniCPM4.1-8B-AutoAWQ)）或 GPTQ（[`openbmb/MiniCPM4.1-8B-GPTQ`](https://huggingface.co/openbmb/MiniCPM4.1-8B-GPTQ)）。

## 6. 注意事项

- 必须传 `--trust-remote-code`，因为建模代码是从 Hub 动态加载的。
- 不要把 `--max-model-len` 设得过高，vLLM 会按这个长度预分配 KV 缓存。
- 音频 / 图像输入请使用 MiniCPM-V 或 MiniCPM-o 多模态模型；MiniCPM 4.1 为纯文本。
