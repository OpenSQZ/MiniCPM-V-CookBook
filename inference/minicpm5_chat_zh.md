# Chat（MiniCPM 5）

> MiniCPM 5 是纯文本 LLM，HuggingFace 模型卡为 [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B)。它使用**标准 `LlamaForCausalLM` 架构** —— 无需 `trust_remote_code`、无需自定义 kernel —— 并支持**混合思考模式**：同一权重既可以直接给出答案，也可以先输出 `<think>` 思维链再给答案，由请求参数控制。

## 初始化模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openbmb/MiniCPM5-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",      # 现代 GPU / Apple Silicon 上为 bfloat16
    device_map="auto",
).eval()
```

> [!TIP]
> MiniCPM5-1B 使用 transformers 上游的 `LlamaForCausalLM`，**不需要 `trust_remote_code=True`**。请确保 `transformers>=5.6`（CUDA 12.x 驱动主机可回退使用 `transformers==4.57.3`）。

## 直接回答（关闭思考）

快速模式直接返回最终答案，不包含 `<think>` 块 —— 适合常规对话、摘要、检索增强问答等场景。

```python
messages = [{"role": "user", "content": "你是谁？请简要介绍一下你自己。"}]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## 混合思考（开启思考）

将 `enable_thinking=True` 后，MiniCPM 5 会先输出 `<think>...</think>` 思维链，再给出最终答案。适合数学、代码、多步规划等需要显式推理的任务。

```python
messages = [{"role": "user", "content": "鸡兔同笼，头共 10 个，脚共 28 只，问鸡和兔各几只？"}]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

典型输出形如：

```text
<think>
设鸡的数量为 x，兔的数量为 y。则：x + y = 10，2x + 4y = 28。
由 x = 10 - y 代入：2(10 - y) + 4y = 28 → y = 4，x = 6。
</think>
鸡有 6 只，兔有 4 只。
```

> [!TIP]
> 采样超参对推理质量影响较大。OpenBMB 官方推荐：开启思考时 `temperature=0.9, top_p=0.95`；关闭思考时 `temperature=0.7, top_p=0.95`。模型自带的 `generation_config.json` 默认按 **Think** 模式调优。

## 多轮对话

`apply_chat_template` 本身无状态 —— 自行维护 `messages` 列表，每轮传入完整历史。思考开关是**逐请求生效**的，同一对话不同轮可以混用。

```python
messages = [
    {"role": "user", "content": "请为第一次来北京的游客规划 3 天行程。"},
]

def chat(messages, enable_thinking=False):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=1024,
        do_sample=True, temperature=0.7, top_p=0.95,
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

reply = chat(messages)
print(reply)

messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "估算一下人均预算。"})
print(chat(messages, enable_thinking=True))    # 预算估算开启思考
```

## CPU 推理

模型 1.08B 参数，fp32 < 4.5 GB、bf16 约 2.2 GB，可纯 CPU 运行 —— 笔记本、CI、无 GPU 环境都没问题：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openbmb/MiniCPM5-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,    # 支持 AVX-512 BF16 / AMX 的主机也可用 bf16
    device_map="cpu",
).eval()

messages = [{"role": "user", "content": "用一句话解释什么是 GQA。"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,        # CPU 延迟敏感，推荐 No-Think
    return_dict=True,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## 工具调用（Tool calling）

MiniCPM 5 在 assistant 输出中以 XML 风格发出 tool call。要得到 OpenAI 兼容的 `tool_calls` JSON，**推荐使用 SGLang 后端** —— SGLang 内置 `minicpm5` parser 可原生转换。完整示例参见 [SGLang 部署文档](../deployment/sglang.html)。

如需在原生 `transformers` 中驱动工具调用，把 `tools` 列表传给 `apply_chat_template`，模型输出后自行解析 XML 块即可。chat template 会按 MiniCPM 5 约定注入工具定义与说明。

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## 长上下文推理

MiniCPM 5 原生支持 **131,072 token** 上下文。输入超过约 32K 时建议启用 Flash Attention 2 以节省显存：

```python
model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM5-1B",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
    device_map="auto",
).eval()
```

## 推荐采样参数

| 模式 | `enable_thinking` | `temperature` | `top_p` | 适用场景 |
| :--- | :---: | :---: | :---: | :--- |
| **Think** | `True` | 0.9 | 0.95 | 推理、数学、代码、多步规划 |
| **No-Think** | `False` | 0.7 | 0.95 | 快速对话、延迟敏感场景 |

## 注意事项

- **不需要** `trust_remote_code=True` —— MiniCPM 5 是标准 `LlamaForCausalLM`，可直接通过 `AutoModelForCausalLM` 加载。
- `enable_thinking` 是 chat template 变量；如果你自己拼 prompt，开启思考时给 assistant 段前缀加上 `<think>\n`，关闭时留空即可。
- 服务化与工具调用更详细的方案参见本版本下的部署指南：vLLM、SGLang（含 `minicpm5` parser）、llama.cpp、Ollama、MLX。
