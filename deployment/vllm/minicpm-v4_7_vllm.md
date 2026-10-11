# MiniCPM-V 4.7 — vLLM Deployment Guide

MiniCPM-V 4.7 ships in **two sizes**, both natively supported by vLLM:

| Size                  | HuggingFace ID                                                                                   | ModelScope ID                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| 1B (dense)            | [openbmb/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)                       | [OpenBMB/MiniCPM-V-4.7-1B](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B)                  |
| 35B-A3B (MoE)         | [openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)             | [OpenBMB/MiniCPM-V-4.7-35B-A3B](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B)        |

> MiniCPM-V 4.7 is registered as `MiniCPMV4_7ForConditionalGeneration` and was merged into the
> official vLLM `main` branch in [PR #58674](https://github.com/vllm-project/vllm/pull/58674).
> **No fork is required** — install vLLM from upstream and make sure your checkout is at or after
> that commit. vLLM requires **`transformers >= 5.18.0`**, where 4.7 is a native architecture.

## 1. Environment Setup

```bash
# Create a clean conda environment
conda create -n vllm-v47 python=3.10 -y
conda activate vllm-v47

# Install from upstream main (requires CUDA toolkit; HEAD must include #58674)
git clone https://github.com/vllm-project/vllm.git
cd vllm
MAX_JOBS=6 VLLM_USE_PRECOMPILED=1 pip install --editable . -v
```

> [!TIP]
> Once a vLLM release containing #58674 is published, `pip install -U vllm` is enough.

For video inference, install the video module:

```bash
pip install vllm[video]
```

Verify the installation:

```bash
python -c "import vllm, transformers; print('vllm', vllm.__version__, '| transformers', transformers.__version__)"
```

## 2. API Service Deployment

### 2.1 Launch API Service

```bash
vllm serve <model_path> \
  --dtype auto \
  --max-model-len 8192 \
  --api-key token-abc123 \
  --gpu_memory_utilization 0.9 \
  --max-num-batched-tokens 8192
```

- `<model_path>`: local path, or `openbmb/MiniCPM-V-4.7-1B` / `openbmb/MiniCPM-V-4.7-35B-A3B`.
  **No `--trust-remote-code` needed** — the architecture is native to `transformers >= 5.18.0`.
- `--api-key`: API access key.
- `--max-model-len`: maximum context length (4.7 supports up to 256K; start small to fit memory).
- `--gpu_memory_utilization`: GPU memory utilization rate.

### 2.2 Image Inference

```python
from openai import OpenAI
import base64

openai_api_key = "token-abc123"  # must match the value passed to `vllm serve`
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

with open('./assets/airplane.jpeg', 'rb') as file:
    image = "data:image/jpeg;base64," + base64.b64encode(file.read()).decode('utf-8')

chat_response = client.chat.completions.create(
    model="<model_path>",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Please describe this image"},
            {"type": "image_url", "image_url": {"url": image}},
        ],
    }],
    extra_body={
        # 4.7 uses the Qwen3.5 backbone — same stop token IDs as 4.6
        "stop_token_ids": [248044, 248046],
        # disable the <think> block for plain replies
        "chat_template_kwargs": {"enable_thinking": False},
    }
)

print(chat_response.choices[0].message.content)
```

### 2.3 Video Inference

```python
from openai import OpenAI
import base64

openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

with open('./videos/video.mp4', 'rb') as video_file:
    video_base64 = base64.b64encode(video_file.read()).decode('utf-8')

chat_response = client.chat.completions.create(
    model="<model_path>",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe this video"},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_base64}"}},
            ],
        },
    ],
    extra_body={"stop_token_ids": [248044, 248046]},
)
print(chat_response.choices[0].message.content)
```

To raise/lower the per-video frame slice budget, pass `mm_processor_kwargs`:

```python
extra_body = {
    "stop_token_ids": [248044, 248046],
    "mm_processor_kwargs": {"max_slice_nums": 2},
}
```

### 2.4 Thinking Mode

4.7 is a single checkpoint per size: the chat template enables `<think>` by default. Disable it
per request with `chat_template_kwargs`:

```python
extra_body = {
    "stop_token_ids": [248044, 248046],
    "chat_template_kwargs": {"enable_thinking": False},
}
```

### 2.5 Multi-image / Multi-video

Raise the `--limit-mm-per-prompt` budget at launch:

```bash
# allow up to 5 images and 2 videos per request
vllm serve <model_path> --dtype auto --max-model-len 16384 --api-key token-abc123 \
  --gpu_memory_utilization 0.9 \
  --limit-mm-per-prompt '{"image": 5, "video": 2}'
```

## 3. Offline Inference

```python
from transformers import AutoProcessor
from PIL import Image
from vllm import LLM, SamplingParams

MODEL_NAME = "<model_path>"  # or "openbmb/MiniCPM-V-4.7-1B"

image = Image.open("./assets/airplane.jpeg").convert("RGB")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

llm = LLM(
    model=MODEL_NAME,
    max_model_len=8192,
    disable_mm_preprocessor_cache=True,
    limit_mm_per_prompt={"image": 5},
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "Please describe the content of this image"},
    ],
}]

prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}

sampling_params = SamplingParams(
    stop_token_ids=[248044, 248046],
    temperature=0.7,
    top_p=0.8,
    max_tokens=1024,
)

outputs = llm.generate(inputs, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

## Notes

1. **Model path**: replace `<model_path>` with a local path or `openbmb/MiniCPM-V-4.7-1B` / `openbmb/MiniCPM-V-4.7-35B-A3B`.
2. **Stop tokens**: `[248044, 248046]` (Qwen3.5 vocab; v4.5 used `[1, 151645]`).
3. **Memory**: the 1B model fits on a single consumer GPU; **35B-A3B** (~70 GB bf16) needs multiple GPUs or 4-bit weights (`-AWQ` / `-GPTQ` / `-FP8` / `-INT4`). Tune `--gpu_memory_utilization`, `--max-model-len`, `--max-num-batched-tokens`.
4. **Multimodal limits**: set `--limit-mm-per-prompt` for multi-image / multi-video sessions.
5. **No `trust_remote_code`**: 4.7 is native in `transformers >= 5.18.0`.
