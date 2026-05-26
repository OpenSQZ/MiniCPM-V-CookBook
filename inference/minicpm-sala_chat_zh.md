# Chat（MiniCPM-SALA）

> MiniCPM-SALA 是探索**稀疏 + 线性混合注意力**的研究版本，模型卡为 [`openbmb/MiniCPM-SALA`](https://huggingface.co/openbmb/MiniCPM-SALA)。它由 MiniCPM 4 通过结构化衰减与后训练适应蒸馏而来，面向百万级令牌输入，计算量仅为密集 Transformer 的一小部分。

## 初始化模型

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

## 基础对话

```python
messages = [{"role": "user", "content": "请总结这篇文章：..."}]

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

## 超长文档推理

SALA 的 25% 稀疏 + 75% 线性混合注意力专为超长输入设计（≥ 128K 令牌，配合 [HyPE](https://arxiv.org/abs/2601.22156) 位置编码可扩展到约 1M 令牌）。读取整篇文档后作为单条 user 消息传入即可——线性注意力层让 KV 缓存占用保持有界。

```python
with open("long_document.txt", "r", encoding="utf-8") as f:
    long_text = f.read()

messages = [{
    "role": "user",
    "content": f"以下是一篇长文档，请阅读后回答末尾的问题。\n\n{long_text}\n\n问题：第三节的主要观点是什么？"
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
> SALA 继承自 MiniCPM 4 的 tokenizer，prompt 模板与 4 相同。实际差异是可以容纳长一个数量级的输入而不爆显存。

## 注意事项

- SALA 是**研究版本**，API 与行为后续可能演进。架构细节参见[技术报告（PDF）](https://github.com/OpenBMB/MiniCPM/blob/main/docs/MiniCPM_SALA.pdf)。
- 稀疏注意力层依赖 [InfLLM-V2 CUDA kernel](https://github.com/OpenBMB/infllmv2_cuda_impl)。HF Transformers 推理时会自动回落到较慢的参考实现；高吞吐服务请安装 kernel 并使用本版本下的 SGLang 指南。
- SALA 的 SGLang 推理需要使用 [`tc-mb/sglang`](https://github.com/tc-mb/sglang) fork 的 `minicpm_sala` 分支 —— 详见本节 `Deployment > SGLang`。
