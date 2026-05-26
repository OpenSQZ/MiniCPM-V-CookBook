# MiniCPM 5 - vLLM

> [!NOTE]
> MiniCPM 5 已在 **vLLM 上游主分支**直接支持，无需 fork。由于模型本身就是标准 `LlamaForCausalLM`，vLLM `>= 0.21` 可直接加载 checkpoint —— 无自定义 kernel、无需 `trust_remote_code`。是生产级高吞吐 + OpenAI 兼容 chat completions 的推荐路径。

## 1. 环境准备

```bash
pip install "vllm>=0.21"          # 最新版（CUDA 13.x 驱动主机）
# pip install "vllm==0.10.1.1"    # CUDA 12.x 驱动主机回退版本
```

> [!TIP]
> MiniCPM 5 需要 vLLM `>= 0.21`。更早版本不包含标准 `LlamaForCausalLM` 的最新集成路径。

## 2. API 服务

### 启动 server

```bash
vllm serve openbmb/MiniCPM5-1B \
    --served-model-name MiniCPM5-1B \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

服务默认在 `http://localhost:8000` 暴露 OpenAI 兼容的 `/v1/chat/completions` 接口。

### 关键参数

| 参数 | 默认值 | 何时调整 |
| :--- | :--- | :--- |
| `--max-model-len` | `131072`（128K 原生上下文） | 小显存 GPU 可降到 `8192` / `32768` 来释放 KV-cache |
| `--gpu-memory-utilization` | `0.85` | **共享 GPU** 上要降低 —— vLLM 在 `(空闲 / 总显存) < 该值` 时会硬失败 |
| `--dtype` | `bfloat16` | 老 GPU 用 `float16`（新 NVIDIA 优先 bf16） |
| `--enforce-eager` | 未设置 | 极小显存场景 CUDA graph OOM 时可启用 |

### 调用服务（OpenAI client）

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

resp = client.chat.completions.create(
    model="MiniCPM5-1B",
    messages=[{"role": "user", "content": "写一篇关于端侧 AI 的短文。"}],
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### 通过 API 启用混合思考

通过 `extra_body.chat_template_kwargs.enable_thinking` 在单次请求中切换 `<think>` 推理模式：

```python
resp = client.chat.completions.create(
    model="MiniCPM5-1B",
    messages=[{"role": "user", "content": "鸡兔同笼，头共 10 个，脚共 28 只，问鸡和兔各几只？"}],
    temperature=0.9,
    top_p=0.95,
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
print(resp.choices[0].message.content)
```

| 模式 | `enable_thinking` | `temperature` | `top_p` |
| :--- | :---: | :---: | :---: |
| Think | `true` | 0.9 | 0.95 |
| No-Think | `false` | 0.7 | 0.95 |

### 快速验证（curl）

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "max_tokens": 64
    }'
```

## 3. 离线 / 批量推理

```python
from vllm import LLM, SamplingParams

llm = LLM(model="openbmb/MiniCPM5-1B", dtype="bfloat16", max_model_len=131072)
out = llm.chat(
    [[{"role": "user", "content": "用一句话解释 GQA。"}]],
    SamplingParams(temperature=0.9, top_p=0.95, max_tokens=512),
    chat_template_kwargs={"enable_thinking": True},
)
print(out[0].outputs[0].text)
```

## 4. 工具调用（Tool calling）

需要原生 `tool_calls` JSON 输出请优先选择 [SGLang 部署文档](sglang.html) —— SGLang 内置 `minicpm5` parser。在 vLLM 下，MiniCPM 5 的 XML 风格 tool call 会落在 assistant `content` 字符串中，需要客户端自行解析。

## 5. 注意事项

- MiniCPM 5 为纯文本模型；图像 / 音频输入请使用 MiniCPM-V 或 MiniCPM-o。
- **不需要** `--trust-remote-code` —— MiniCPM 5 是标准 `LlamaForCausalLM`。
- `--max-model-len` 不应超过原生 131,072 token。vLLM 会按完整上下文预分配 KV cache，所以请按业务需求设到尽量小的值。
