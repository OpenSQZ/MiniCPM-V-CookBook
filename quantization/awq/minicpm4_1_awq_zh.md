# MiniCPM 4.1 - AWQ

> [!NOTE]
> 官方已发布 AWQ 量化权重 [`openbmb/MiniCPM4.1-8B-AutoAWQ`](https://huggingface.co/openbmb/MiniCPM4.1-8B-AutoAWQ)。大多数用户直接下载即可，无需自行 calibration。

## 方法 1 —— 使用预量化模型

### 下载

```bash
git clone https://huggingface.co/openbmb/MiniCPM4.1-8B-AutoAWQ
```

ModelScope 镜像：<https://www.modelscope.cn/models/OpenBMB/MiniCPM4.1-8B-AutoAWQ>

### vLLM 推理

```bash
pip install -U vllm
vllm serve openbmb/MiniCPM4.1-8B-AutoAWQ --trust-remote-code --max-model-len 65536
```

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
print(client.chat.completions.create(
    model="openbmb/MiniCPM4.1-8B-AutoAWQ",
    messages=[{"role": "user", "content": "写一篇关于端侧 AI 的短文。"}],
).choices[0].message.content)
```

vLLM 在 Ampere / Ada / Hopper 上默认走 [AWQ-Marlin](https://github.com/IST-DASLab/marlin) INT4 kernel，因此 AWQ checkpoint 的吞吐通常与纯 Marlin checkpoint 相当甚至更快。

### Transformers 推理

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM4.1-8B-AutoAWQ", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM4.1-8B-AutoAWQ",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()
```

> [!IMPORTANT]
> Transformers 推理需要安装 AWQ kernel：`pip install autoawq`。Transformers `>= 4.43` 会自动加载 AWQ runtime。

## 方法 2 —— 自己量化

如果需要自定义 calibration set，可以自行运行 AWQ。

### 安装 AutoAWQ

```bash
pip install autoawq
```

> 部分 MiniCPM 变体需要 [`tc-mb/AutoAWQ`](https://github.com/tc-mb/AutoAWQ) fork 中的 kernel 补丁。MiniCPM 4.1 目前使用上游 `autoawq` 即可。

### 量化脚本

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path  = "openbmb/MiniCPM4.1-8B"
output_path = "./MiniCPM4.1-8B-AWQ"

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(output_path)
tokenizer.save_pretrained(output_path)
```

输出目录就是标准 HF 模型，加载方式与官方 `openbmb/MiniCPM4.1-8B-AutoAWQ` 完全一致。

## 注意事项

- AWQ Group Size `128`、W-bit `4`、ZP 启用 —— 与官方发布对齐的配置。
- 在 Ampere/Ada/Hopper 上单 token 延迟敏感的部署中，AWQ + Marlin kernel 在质量接近 FP16 的同时吞吐约为 FP16 的 2 倍。
- 0.5B 模型没有 AWQ 版本。受限端侧请通过 `llama.cpp` 使用 GGUF Q4。
