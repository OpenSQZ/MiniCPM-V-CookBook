# Chat (MiniCPM 4.1)

> MiniCPM 4.1 is a text-only LLM available on HuggingFace as [`openbmb/MiniCPM4.1-8B`](https://huggingface.co/openbmb/MiniCPM4.1-8B). It ships with **hybrid reasoning** — a single checkpoint that can produce direct answers or step-by-step `<think>` chains depending on a per-request flag.

## Initialize model

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

## Direct chat (reasoning OFF)

The default behaviour returns a final answer without any `<think>` block — best for straightforward instructions, summarisation, retrieval-augmented chat, etc.

```python
messages = [{"role": "user", "content": "Write a short article about Artificial Intelligence."}]

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

## Hybrid reasoning (reasoning ON)

Toggle `enable_thinking=True` to make MiniCPM 4.1 first emit a `<think>...</think>` chain-of-thought block, then the final answer. Recommended for math, code, multi-step planning and other tasks that benefit from explicit deliberation.

```python
messages = [{"role": "user", "content": "If a train leaves city A at 9:00 and travels at 80 km/h to city B 200 km away, when does it arrive?"}]

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

A typical reply has the shape:

```text
<think>
Distance is 200 km, speed is 80 km/h, so the trip takes 200 / 80 = 2.5 hours.
9:00 + 2 hours 30 minutes = 11:30.
</think>
The train arrives at 11:30.
```

> [!TIP]
> Sampling hyper-parameters affect reasoning quality. The OpenBMB team recommends `temperature=0.6, top_p=0.95` for reasoning-on, and `temperature=0.7, top_p=0.8` for reasoning-off.

## Multi-turn conversation

`apply_chat_template` is stateless — keep the running `messages` list yourself and pass it back in on every turn. The reasoning toggle is **per request**, so different turns can use different modes.

```python
messages = [
    {"role": "user", "content": "Plan a 3-day Beijing trip for a first-time visitor."},
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
messages.append({"role": "user", "content": "Estimate the budget per person."})
print(chat(messages, enable_thinking=True))
```

## Long-context inference

MiniCPM 4.1 supports up to **128K** context via [InfLLM-V2](https://arxiv.org/abs/2509.24663) sparse attention. For inputs longer than ~32K, switch on `flash_attention_2` for memory savings.

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
).eval().cuda()
```

## Notes

- `trust_remote_code=True` is required — MiniCPM 4.1 ships its modelling code on the Hub and is **not yet merged into upstream `transformers`** at the time of writing.
- `enable_thinking` is a chat-template variable; if you build your own prompt manually, prepend `<think>\n` to the assistant turn for reasoning-on and leave the assistant prefix empty otherwise.
- The 0.5B variant ([`openbmb/MiniCPM4-0.5B`](https://huggingface.co/openbmb/MiniCPM4-0.5B)) shares the same chat template — drop-in for tightly constrained edge deployments.
