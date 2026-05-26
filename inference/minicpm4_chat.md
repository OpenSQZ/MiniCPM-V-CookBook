# Chat (MiniCPM 4)

> MiniCPM 4 is the previous-generation text LLM in the MiniCPM family, available as [`openbmb/MiniCPM4-8B`](https://huggingface.co/openbmb/MiniCPM4-8B) and the ultra-light [`openbmb/MiniCPM4-0.5B`](https://huggingface.co/openbmb/MiniCPM4-0.5B). For new projects we recommend [MiniCPM 4.1](../minicpm4_1/overview.html); this guide covers the original 4 release.

## Initialize model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM4-8B"   # or openbmb/MiniCPM4-0.5B

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()
```

## Basic chat

MiniCPM 4 does **not** ship the hybrid-reasoning toggle that 4.1 has — every reply is a direct answer. Use `apply_chat_template` to assemble the prompt.

```python
messages = [{"role": "user", "content": "Write an article about Artificial Intelligence."}]

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

## Multi-turn conversation

`apply_chat_template` is stateless — keep the running `messages` list yourself and pass it back each turn.

```python
messages = [{"role": "user", "content": "Plan a 3-day Beijing trip for a first-time visitor."}]

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
messages.append({"role": "user", "content": "Estimate the budget per person."})
print(chat(messages))
```

## Long-context inference

MiniCPM 4 introduced [InfLLM-V2](https://arxiv.org/abs/2509.24663) sparse attention for **128K** context. For sequences beyond ~32K, switch on Flash Attention 2.

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
).eval().cuda()
```

## Notes

- `trust_remote_code=True` is required — MiniCPM 4 ships its modelling code on the Hub.
- The 0.5B variant shares the same chat template — recommended for tightly constrained edge deployments where the 8B doesn't fit.
- For accelerated generation, see the `Deployment` section under this version (vLLM with EAGLE / Marlin).
