# Chat (MiniCPM-SALA)

> MiniCPM-SALA is a research release exploring **sparse + linear hybrid attention** for ultra-long contexts. Available as [`openbmb/MiniCPM-SALA`](https://huggingface.co/openbmb/MiniCPM-SALA). Distilled from MiniCPM 4 with structured decay and post-training adaptation, it targets million-token inputs at a fraction of the dense-Transformer compute.

## Initialize model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "openbmb/MiniCPM-SALA"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()
```

## Basic chat

```python
messages = [{"role": "user", "content": "Summarise this article: ..."}]

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

## Long-document inference

SALA's hybrid 25% sparse + 75% linear attention is designed for very long inputs (≥ 128K tokens, scalable to ~1M tokens with [HyPE](https://arxiv.org/abs/2601.22156) positional encoding). Read the file and pass the whole thing as one user message — the model's KV-cache footprint stays bounded thanks to the linear-attention layers.

```python
with open("long_document.txt", "r", encoding="utf-8") as f:
    long_text = f.read()

messages = [{
    "role": "user",
    "content": f"Below is a long document. Read it and answer the question that follows.\n\n{long_text}\n\nQuestion: What is the main argument of section 3?"
}]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

out_ids = model.generate(input_ids, max_new_tokens=1024, do_sample=True,
                         temperature=0.7, top_p=0.8)
print(tokenizer.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True))
```

> [!TIP]
> SALA inherits its tokenizer from MiniCPM 4, so the same prompt template works. The main practical difference is that you can feed inputs an order of magnitude longer without OOM.

## Notes

- SALA is a **research checkpoint** — APIs and behaviour may evolve. Refer to the [technical report (PDF)](https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf) for architectural details.
- The sparse-attention layers depend on [InfLLM-V2 CUDA kernels](https://github.com/OpenBMB/infllmv2_cuda_impl). For HF Transformers inference the model auto-falls-back to a slower reference path; for high-throughput serving install the kernels and use the SGLang guide under this version.
- SGLang inference for SALA requires the [`tc-mb/sglang`](https://github.com/tc-mb/sglang) fork on the `minicpm_sala` branch — see `Deployment > SGLang` in this section.
