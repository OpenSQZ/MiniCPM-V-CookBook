# Chat（MiniCPM 4.1）

> MiniCPM 4.1 是纯文本 LLM，HuggingFace 模型卡为 [`openbmb/MiniCPM4.1-8B`](https://huggingface.co/openbmb/MiniCPM4.1-8B)。它支持**混合思考模式**：同一权重可以直接给出答案，也可以先输出 `<think>` 思维链再给答案，由请求参数控制。

## 初始化模型

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM4.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()
```

## 直接回答（关闭思考）

默认行为是直接返回最终答案，不包含 `<think>` 块 —— 适合常规指令、摘要、检索增强问答等场景。

```python
messages = [{"role": "user", "content": "写一篇关于人工智能的短文。"}]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False,
).to(model.device)

out_ids = model.generate(
    input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
)
answer = tokenizer.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(answer)
```

## 混合思考（开启思考）

将 `enable_thinking=True` 后，MiniCPM 4.1 会先输出 `<think>...</think>` 思维链，再给出最终答案。适合数学、代码、多步规划等需要显式推理的任务。

```python
messages = [{"role": "user", "content": "某列车 9:00 从 A 城出发，以 80 km/h 行驶到 200 km 外的 B 城，几点到达？"}]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=True,
).to(model.device)

out_ids = model.generate(
    input_ids,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
)
print(tokenizer.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True))
```

典型输出形如：

```text
<think>
距离 200 km，速度 80 km/h，因此用时 200 / 80 = 2.5 小时。
9:00 + 2 小时 30 分 = 11:30。
</think>
列车在 11:30 到达。
```

> [!TIP]
> 采样超参对推理质量影响较大。OpenBMB 官方推荐：开启思考时 `temperature=0.6, top_p=0.95`，关闭思考时 `temperature=0.7, top_p=0.8`。

## 多轮对话

`apply_chat_template` 本身无状态 —— 自行维护 `messages` 列表，每轮传入完整历史。思考开关是**逐请求生效**的，同一对话不同轮可以混用。

```python
messages = [
    {"role": "user", "content": "请为第一次来北京的游客规划 3 天行程。"},
]

def chat(messages, enable_thinking=False):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    ).to(model.device)
    out_ids = model.generate(input_ids, max_new_tokens=1024, do_sample=True,
                             temperature=0.7, top_p=0.8)
    return tokenizer.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

reply = chat(messages)
print(reply)

messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "估算一下人均预算。"})
print(chat(messages, enable_thinking=True))
```

## 长上下文推理

MiniCPM 4.1 通过 [InfLLM-V2](https://arxiv.org/abs/2509.24663) 稀疏注意力支持最长 **128K** 上下文。输入超过约 32K 时建议启用 `flash_attention_2` 以节省显存。

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
).eval().cuda()
```

## 注意事项

- 必须设置 `trust_remote_code=True` —— MiniCPM 4.1 的建模代码托管在 Hub 上，截至当前**尚未合入 `transformers` 上游**。
- `enable_thinking` 是 chat template 变量；如果你自己拼 prompt，开启思考时给 assistant 段前缀加上 `<think>\n`，关闭时留空即可。
- 0.5B 变体 [`openbmb/MiniCPM4-0.5B`](https://huggingface.co/openbmb/MiniCPM4-0.5B) 共用同一 chat template，可作为资源受限端侧场景的替代。
