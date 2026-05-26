# Chat (MiniCPM 5)

> MiniCPM 5 is a text-only LLM available on HuggingFace as [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B). It uses the **standard `LlamaForCausalLM` architecture** — no `trust_remote_code`, no custom kernels — and ships **hybrid reasoning**: a single checkpoint can produce direct answers or step-by-step `<think>` chains depending on a per-request flag.

## Initialize model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openbmb/MiniCPM5-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",      # bfloat16 on modern GPUs / Apple Silicon
    device_map="auto",
).eval()
```

> [!TIP]
> MiniCPM5-1B's modeling code is upstream `LlamaForCausalLM`, so `trust_remote_code=True` is **not required**. Make sure you have `transformers>=5.6` (or the fallback combo `transformers==4.57.3` for CUDA 12.x driver hosts).

## Direct chat (No-Think)

The fast path returns a final answer without any `<think>` block — best for everyday assistant turns, summarisation, retrieval-augmented chat, etc.

```python
messages = [{"role": "user", "content": "Who are you? Please briefly introduce yourself."}]

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

## Hybrid reasoning (Think)

Toggle `enable_thinking=True` to make MiniCPM 5 first emit a `<think>...</think>` chain-of-thought block, then the final answer. Recommended for math, code, multi-step planning and other tasks that benefit from explicit deliberation.

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

A typical reply has the shape:

```text
<think>
Let chickens = x, rabbits = y. We have x + y = 10 and 2x + 4y = 28.
From the first equation x = 10 - y; substituting: 2(10 - y) + 4y = 28 → y = 4, x = 6.
</think>
鸡有 6 只，兔有 4 只。
```

> [!TIP]
> Sampling hyper-parameters affect reasoning quality. The OpenBMB team recommends `temperature=0.9, top_p=0.95` for Think, and `temperature=0.7, top_p=0.95` for No-Think. The `generation_config.json` shipped with the checkpoint is tuned for **Think** mode by default.

## Multi-turn conversation

`apply_chat_template` is stateless — keep the running `messages` list yourself and pass it back in on every turn. The reasoning toggle is **per request**, so different turns can use different modes.

```python
messages = [
    {"role": "user", "content": "Plan a 3-day Beijing trip for a first-time visitor."},
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
messages.append({"role": "user", "content": "Estimate the budget per person."})
print(chat(messages, enable_thinking=True))    # turn on reasoning for the budgeting question
```

## CPU-only inference

The full 1.08B-param model is < 4.5 GB in fp32 / ~2.2 GB in bf16, so it runs on CPU only — laptops, CI runners, no-GPU sanity checks all work:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openbmb/MiniCPM5-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,    # bf16 also works on AVX-512 BF16 / AMX hosts
    device_map="cpu",
).eval()

messages = [{"role": "user", "content": "用一句话解释什么是 GQA。"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,        # No-Think recommended for CPU latency
    return_dict=True,
    return_tensors="pt",
)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Tool calling

MiniCPM 5 emits XML-style tool calls inside its assistant turn. For OpenAI-compatible `tool_calls` JSON, **SGLang is the recommended backend** — it ships a native `minicpm5` parser. See the [SGLang deployment guide](../deployment/sglang.html) for the full recipe.

If you'd rather drive tool calling from raw `transformers`, pass a `tools` argument to `apply_chat_template` and parse the XML block out of the model's output yourself. The chat template knows how to inject the tool definitions and instructions for you.

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
    [{"role": "user", "content": "What is the weather in Beijing?"}],
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

## Long-context inference

MiniCPM 5 natively supports a **131,072-token** context window. For inputs beyond ~32K, switch on Flash Attention 2 to save memory:

```python
model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM5-1B",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
    device_map="auto",
).eval()
```

## Recommended sampling

| Mode | `enable_thinking` | `temperature` | `top_p` | When to use |
| :--- | :---: | :---: | :---: | :--- |
| **Think** | `True` | 0.9 | 0.95 | reasoning, math, code, multi-step planning |
| **No-Think** | `False` | 0.7 | 0.95 | fast assistant, latency-bound chat |

## Notes

- `trust_remote_code=True` is **not required** — MiniCPM 5 ships as a vanilla `LlamaForCausalLM` and loads through `AutoModelForCausalLM` directly.
- `enable_thinking` is a chat-template variable; if you build prompts manually, prepend `<think>\n` to the assistant turn for Think mode and leave the prefix empty otherwise.
- For accelerated serving and tool calling, see the deployment guides under this version (vLLM, SGLang with `minicpm5` parser, llama.cpp, Ollama, MLX).
