# FP8 / INT4 — llm-compressor（MiniCPM-V 4.7）

MiniCPM **35B-A3B** 的 MoE 专家以*融合参数*存储（`mlp.experts.gate_up_proj`、
`mlp.experts.down_proj`）。普通 AutoAWQ / AutoGPTQ / bitsandbytes 只量化 `nn.Linear`，
会把占权重约 92% 的专家留在 bf16。要做可用的 4-bit 35B-A3B，需用支持融合专家的工具，例如
**`llm-compressor`**（产出 `compressed-tensors` 权重）。

已发布的 35B-A3B 仓库 —— `-FP8`、`-INT4`、`-int4`、`-AWQ`、`-GPTQ` —— 都是
`compressed-tensors` 权重，可直接用 vLLM 加载。

## 加载预量化权重

```python
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image

MODEL = "openbmb/MiniCPM-V-4.7-35B-A3B-FP8"  # 或 -INT4 / -int4 / -AWQ / -GPTQ
processor = AutoProcessor.from_pretrained(MODEL)
llm = LLM(model=MODEL, max_model_len=8192, limit_mm_per_prompt={"image": 5})

image = Image.open("example.jpg").convert("RGB")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "描述这张图片"}]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out = llm.generate({"prompt": prompt, "multi_modal_data": {"image": image}},
                   SamplingParams(stop_token_ids=[248044, 248046], max_tokens=1024))
print(out[0].outputs[0].text)
```

## 自行量化

安装 `llm-compressor`（它固定了 `transformers`，建议用独立环境）：

```bash
pip install llmcompressor
```

### 配方 —— FP8（W8A8，动态）

```yaml
# fp8_recipe.yaml
default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore:
        - 're:.*lm_head.*'
        - 're:.*vision_tower.*'
        - 're:.*vit_merger.*'
        - 're:.*merger.*'
        - 're:.*linear_attn.*'
        - 're:.*mlp\.gate$'
        - 're:.*shared_expert.*'
        - 're:.*embed_tokens.*'
      scheme: FP8_DYNAMIC
      requires_calibration_data: false
```

### 配方 —— INT4（W4A16，group 128）

```yaml
# w4a16_recipe.yaml
default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore:
        - 're:.*lm_head.*'
        - 're:.*vision_tower.*'
        - 're:.*vit_merger.*'
        - 're:.*merger.*'
        - 're:.*self_attn.*'
        - 're:.*linear_attn.*'
        - 're:.*mlp\.gate$'
        - 're:.*shared_expert.*'
        - 're:.*embed_tokens.*'
      scheme: W4A16
      requires_calibration_data: false
```

### 运行

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "openbmb/MiniCPM-V-4.7-35B-A3B"
SAVE  = "./MiniCPM-V-4.7-35B-A3B-INT4"

model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head", "re:.*vision_tower.*", "re:.*vit_merger.*", "re:.*merger.*",
            "re:.*self_attn.*", "re:.*linear_attn.*", "re:.*mlp\\.gate$",
            "re:.*shared_expert.*", "re:.*embed_tokens.*"],
)
oneshot(model=model, recipe=recipe)

model.save_pretrained(SAVE, save_compressed=True)
processor.save_pretrained(SAVE)
```

> `ignore` 列表把视觉塔、merger、router/`gate`、shared expert、linear-attention 与 embedding
> 保持在完整精度 —— 量化它们要么破坏 canvas M-RoPE 视觉链路，要么精度损失大而体积收益小。
> 融合的专家 `gate_up_proj` / `down_proj` **会被量化**，这正是 MoE 模型的关键。

各 35B-A3B 量化仓库内的 `recipe.yaml` 即其实际使用的配方。
