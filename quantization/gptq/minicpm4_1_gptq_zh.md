# MiniCPM 4.1 - GPTQ（含 QAT 变体）

> [!NOTE]
> 官方已发布 GPTQ 量化权重 [`openbmb/MiniCPM4.1-8B-GPTQ`](https://huggingface.co/openbmb/MiniCPM4.1-8B-GPTQ)。MiniCPM4-0.5B-QAT 发布也采用 **GPTQ 格式存储**，本指南一并覆盖。

## 方法 1 —— 使用预量化模型

### 下载

```bash
git clone https://huggingface.co/openbmb/MiniCPM4.1-8B-GPTQ
```

ModelScope 镜像：<https://www.modelscope.cn/models/OpenBMB/MiniCPM4.1-8B-GPTQ>

### vLLM 推理

```bash
pip install -U vllm
vllm serve openbmb/MiniCPM4.1-8B-GPTQ --trust-remote-code --max-model-len 65536
```

vLLM 在 Ampere / Ada / Hopper 上自动选择 GPTQ-Marlin kernel 获得最佳吞吐。

### Transformers 推理

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM4.1-8B-GPTQ", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM4.1-8B-GPTQ",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()
```

> [!IMPORTANT]
> Transformers `>= 4.36` 通过 `optimum` 提供 GPTQ runtime。安装：`pip install optimum auto-gptq`。

## 方法 2 —— 自己量化

```bash
git clone https://github.com/tc-mb/AutoGPTQ.git
cd AutoGPTQ
pip install -e .
```

> `tc-mb/AutoGPTQ` fork 含针对 MiniCPM 的配置补丁。我们已经把 model_type 映射推到 AutoGPTQ-NEXT —— 新项目也可以尝试官方推荐的 [`vllm-project/llm-compressor`](https://github.com/vllm-project/llm-compressor)。

完整量化脚本参考 [`tc-mb/AutoGPTQ` README](https://github.com/tc-mb/AutoGPTQ)，与官方 `MiniCPM4.1-8B-GPTQ` 发布使用同一配方。

## QAT 变体

MiniCPM 4 额外发布了**量化感知训练**变体：[`openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format`](https://huggingface.co/openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format)。

- QAT 模型在训练过程中即引入量化，INT4 权重恢复的精度比训练后 GPTQ 更高。
- 物理上**以 GPTQ 格式存储**，因此 vLLM / Transformers 加载方式完全相同：

```bash
vllm serve openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format \
    --trust-remote-code --max-model-len 32768
```

```python
model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()
```

## 注意事项

- vLLM 中的 GPTQ 在支持的 GPU 上默认走 GPTQ-Marlin kernel，无需额外参数。
- MiniCPM 4（非 4.1）的官方 GPTQ 是 [`openbmb/MiniCPM4-8B-GPTQ`](https://huggingface.co/openbmb/MiniCPM4-8B-GPTQ)，流程一致。
- 内存极度紧张的部署，建议优先通过 `llama.cpp` 使用 GGUF Q4 / Q3，而不是 GPTQ INT4。
