# Chat（MiniCPM 4）

> MiniCPM 4 是 MiniCPM LLM 系列上一代版本，模型卡为 [`openbmb/MiniCPM4-8B`](https://huggingface.co/openbmb/MiniCPM4-8B) 和超轻量的 [`openbmb/MiniCPM4-0.5B`](https://huggingface.co/openbmb/MiniCPM4-0.5B)。新项目推荐使用 [MiniCPM 4.1](../minicpm4_1/overview.html)；本指南覆盖原版 4。

## 初始化模型

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM4-8B"   # 或 openbmb/MiniCPM4-0.5B

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()
```

## 基础对话

MiniCPM 4 **不支持** 4.1 的混合思考开关，每次回复都是直接答案。使用 `apply_chat_template` 组装 prompt。

```python
messages = [{"role": "user", "content": "写一篇关于人工智能的文章。"}]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

out_ids = model.generate(
    input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
)
print(tokenizer.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True))
```

## 多轮对话

`apply_chat_template` 本身无状态，自行维护 `messages` 列表逐轮拼接历史即可。

```python
messages = [{"role": "user", "content": "请为第一次来北京的游客规划 3 天行程。"}]

def chat(messages):
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    out_ids = model.generate(input_ids, max_new_tokens=1024, do_sample=True,
                             temperature=0.7, top_p=0.8)
    return tokenizer.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

reply = chat(messages)
print(reply)

messages.append({"role": "assistant", "content": reply})
messages.append({"role": "user", "content": "估算一下人均预算。"})
print(chat(messages))
```

## 长上下文推理

MiniCPM 4 首次引入 [InfLLM-V2](https://arxiv.org/abs/2509.24663) 稀疏注意力，支持 **128K** 上下文。序列超过约 32K 时建议启用 Flash Attention 2。

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
).eval().cuda()
```

## 注意事项

- 必须设置 `trust_remote_code=True` —— MiniCPM 4 建模代码托管在 Hub 上。
- 0.5B 变体共用同一 chat template，适合 8B 装不下的端侧场景。
- 想要加速生成请查看本版本下的 `Deployment` 章节（vLLM + EAGLE / Marlin）。
