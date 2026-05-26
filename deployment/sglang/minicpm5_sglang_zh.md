# MiniCPM 5 - SGLang

> [!NOTE]
> [SGLang](https://github.com/sgl-project/sglang) `>= 0.5.12` 已原生支持 MiniCPM 5，并内置 **`minicpm5` tool-call parser**，可将模型输出的 XML 风格 tool call 自动转换为 OpenAI 兼容的 `tool_calls` JSON。SGLang 把 MiniCPM 5 当作标准 `LlamaForCausalLM` 加载，提供 **RadixAttention 前缀缓存**、高并发和 OpenAI 兼容 API。**强烈推荐用于工具调用 / function calling 场景。**

## 1. 环境准备

```bash
pip install "sglang[srt]>=0.5.12"          # 最新版（CUDA 13.x 驱动）
# pip install "sglang==0.5.6.post3"        # CUDA 12.x 驱动回退版本
```

> [!TIP]
> SGLang 默认 attention backend 依赖 FlashInfer。安装失败时手动安装：
>
> ```bash
> pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/
> ```

推荐运行时环境变量：

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1
```

## 2. 启动服务

```bash
python -m sglang.launch_server \
    --model-path openbmb/MiniCPM5-1B \
    --served-model-name MiniCPM5-1B \
    --dtype bfloat16 \
    --context-length 131072 \
    --mem-fraction-static 0.85 \
    --tool-call-parser minicpm5 \
    --host 0.0.0.0 \
    --port 30000
```

服务在 `http://localhost:30000/v1` 暴露 OpenAI 兼容接口。

### 关键参数

| 参数 | 默认值 | 何时调整 |
| :--- | :--- | :--- |
| `--context-length` | `131072`（128K 原生） | 小显存 / 共享 GPU 上降低 |
| `--mem-fraction-static` | `0.85` | 共享 GPU 上降低 |
| `--dtype` | `bfloat16` | Ampere 或更老的 GPU 用 `float16` |
| `--tool-call-parser` | 未设置 | 设为 `minicpm5`（或 `auto`）启用原生 tool calls |

## 3. 调用服务

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")

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

## 4. 工具调用（推荐）

MiniCPM 5 输出 XML 风格的 tool call。带上 `--tool-call-parser minicpm5` 后，SGLang 会自动转成 OpenAI 兼容的 `tool_calls` JSON —— 任何 OpenAI 工具客户端可直接接入。

```bash
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-1B",
        "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
        "tool_choice": "auto",
        "temperature": 0.7,
        "max_tokens": 256
    }'
```

返回值中带有标准 `tool_calls` 字段：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Beijing\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

> [!TIP]
> `--tool-call-parser auto` 也能根据 chat template 自动选中 `minicpm5` parser。出于可复现部署考虑，建议显式指定 `--tool-call-parser minicpm5`。

## 5. 离线 / 批量推理（Engine API）

```python
import sglang as sgl

llm = sgl.Engine(
    model_path="openbmb/MiniCPM5-1B",
    tp_size=1,
    mem_fraction_static=0.8,
    context_length=131072,
)

outputs = llm.generate(
    ["用一句话解释什么是 GQA。"],
    sampling_params={
        "temperature": 0.9, "top_p": 0.95, "max_new_tokens": 1024,
        "skip_special_tokens": False,
    },
)
print(outputs)
```

## 6. 注意事项

- MiniCPM 5 已在 **SGLang 上游 `>= 0.5.12`** 直接支持，无需 fork。
- `minicpm5` tool-call parser 是 SGLang 内置组件，无需额外安装。
- 多 GPU 推理添加 `--tp 2`。1B 级模型在现代 GPU 上一般无需 TP > 1。
