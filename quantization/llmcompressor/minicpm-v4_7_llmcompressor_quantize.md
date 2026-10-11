# FP8 / INT4 — llm-compressor (MiniCPM-V 4.7)

MiniCPM-V **35B-A3B** stores its MoE experts as *fused* parameters
(`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`). Plain AutoAWQ / AutoGPTQ / bitsandbytes
only quantize `nn.Linear` modules, so they leave the experts (~92% of the weights) in bf16.
Fused-expert-aware tooling such as **`llm-compressor`** (producing `compressed-tensors`
checkpoints) is required for a usable 4-bit 35B-A3B.

The published 35B-A3B repositories — `-FP8`, `-INT4`, `-int4`, `-AWQ`, `-GPTQ` — are all
`compressed-tensors` checkpoints and load directly with vLLM.

## Load a pre-quantized checkpoint

```python
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image

MODEL = "openbmb/MiniCPM-V-4.7-35B-A3B-FP8"  # or -INT4 / -int4 / -AWQ / -GPTQ
processor = AutoProcessor.from_pretrained(MODEL)
llm = LLM(model=MODEL, max_model_len=8192, limit_mm_per_prompt={"image": 5})

image = Image.open("example.jpg").convert("RGB")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image"}]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out = llm.generate({"prompt": prompt, "multi_modal_data": {"image": image}},
                   SamplingParams(stop_token_ids=[248044, 248046], max_tokens=1024))
print(out[0].outputs[0].text)
```

## Quantize yourself

Install `llm-compressor` (it pins `transformers`; use a dedicated environment):

```bash
pip install llmcompressor
```

### Recipe — FP8 (W8A8, dynamic)

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

### Recipe — INT4 (W4A16, group 128)

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

### Run

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

> The `ignore` list keeps the vision tower, merger, router/`gate`, shared expert, linear-attention,
> and embeddings at full precision — quantizing them either breaks the canvas M-RoPE vision path
> or costs accuracy with little size benefit. The fused expert `gate_up_proj` / `down_proj` **are**
> quantized, which is the whole point for the MoE model.

See the `recipe.yaml` shipped in each 35B-A3B quant repository for the exact recipe used.
