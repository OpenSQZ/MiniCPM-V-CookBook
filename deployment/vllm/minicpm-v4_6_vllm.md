# MiniCPM-V 4.6 vLLM Deployment Guide

MiniCPM-V 4.6 ships as **two separate checkpoints**:

| Variant       | HuggingFace ID                                                                                   | ModelScope ID                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Instruct      | [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6)                            | [OpenBMB/MiniCPM-V-4.6](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6)                       |
| Thinking   | [openbmb/MiniCPM-V-4.6-Thinking](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking)                | [OpenBMB/MiniCPM-V-4.6-Thinking](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-Thinking)           |

> Unlike v4.5 (which switched modes via `enable_thinking`), v4.6 ships think and instruct as **independent checkpoints** — pick the one that matches your use case.

## 1. Environment Setup

### 1.1 Install vLLM

MiniCPM-V 4.6 is supported on the official vLLM `main` branch since [PR #43213](https://github.com/vllm-project/vllm/pull/43213) (merged 2026-05-22). **No fork is required** — install directly from the upstream repo, making sure your checkout is at or after that commit:

```bash
# Create a clean conda environment
conda create -n vllm-v46 python=3.10 -y
conda activate vllm-v46

# Install from upstream main (requires CUDA toolkit; HEAD must include #43213)
git clone https://github.com/vllm-project/vllm.git
cd vllm
MAX_JOBS=6 VLLM_USE_PRECOMPILED=1 pip install --editable . -v
```

> [!TIP]
> Once a vLLM release containing #43213 is published, you can simply `pip install -U vllm` and skip the source build. Watch the [vLLM Releases page](https://github.com/vllm-project/vllm/releases) for updates.

For video inference, install the video module:

```bash
pip install vllm[video]
```

vLLM requires `transformers>=5.7.0`, in which MiniCPM-V 4.6 has been merged as a standalone architecture (`MiniCPMV4_6ForConditionalGeneration`).

You can verify the installation with:

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
  --trust-remote-code \
  --max-num-batched-tokens 8192
```

**Parameter Description:**
- `<model_path>`: Local path to MiniCPM-V-4.6, or the HuggingFace ID `openbmb/MiniCPM-V-4.6` / `openbmb/MiniCPM-V-4.6-Thinking`
- `--api-key`: API access key
- `--max-model-len`: Maximum context length. v4.6 supports up to 256K, but start small to fit GPU memory
- `--gpu_memory_utilization`: GPU memory utilization rate

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
        # v4.6 uses Qwen3.5 backbone with new vocab — note the different stop token IDs
        "stop_token_ids": [248044, 248046]
    }
)

print("Chat response:", chat_response)
print("Chat response content:", chat_response.choices[0].message.content)
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
    extra_body={"stop_token_ids": [248044, 248046]}
)

print(chat_response.choices[0].message.content)
```

> v4.6 keeps video pipeline simple: the image processor is invoked frame-by-frame, and the per-video frame budget is controlled by `mm_processor_kwargs`/`video_processor.max_slice_nums`. No special video backend is required.

To raise/lower the per-video frame slice budget, pass `mm_processor_kwargs`:

```python
extra_body = {
    "stop_token_ids": [248044, 248046],
    "mm_processor_kwargs": {"max_slice_nums": 2},
}
```

### 2.4 Thinking Mode

If you serve the **`openbmb/MiniCPM-V-4.6-Thinking`** checkpoint, the chat template injects a `<think>` block by default and the assistant returns reasoning followed by `</think>` then the final answer. You can shortcut the template via `chat_template_kwargs`:

```python
extra_body = {
    "stop_token_ids": [248044, 248046],
    # disable the leading <think> block on a Thinking model
    "chat_template_kwargs": {"enable_thinking": False},
}
```

The Instruct checkpoint never emits `<think>` blocks — pick the right checkpoint instead of toggling at request time.

### 2.5 Multi-turn Conversation

#### Launch Parameter Configuration

For multi-image / multi-video conversations, raise the `--limit-mm-per-prompt` budget at launch:

```bash
# allow up to 3 videos per request
vllm serve <model_path> --dtype auto --max-model-len 16384 --api-key token-abc123 \
  --gpu_memory_utilization 0.9 --trust-remote-code \
  --limit-mm-per-prompt '{"video": 3}'
```

```bash
# mixed images + video
vllm serve <model_path> --dtype auto --max-model-len 16384 --api-key token-abc123 \
  --gpu_memory_utilization 0.9 --trust-remote-code \
  --limit-mm-per-prompt '{"image": 5, "video": 2}'
```

#### Multi-turn Conversation Example

```python
from openai import OpenAI
import base64
import mimetypes
import os

openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

messages = [{"role": "system", "content": "You are a helpful assistant."}]

def file_to_base64(file_path):
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_mime_type(file_path):
    mime, _ = mimetypes.guess_type(file_path)
    return mime or 'application/octet-stream'

def build_file_content(file_path):
    mime_type = get_mime_type(file_path)
    base64_data = file_to_base64(file_path)
    url = f"data:{mime_type};base64,{base64_data}"
    if mime_type.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": url}}
    elif mime_type.startswith("video/"):
        return {"type": "video_url", "video_url": {"url": url}}
    else:
        print(f"Unsupported file type: {mime_type}")
        return None

while True:
    user_text = input("Please enter your question (type 'exit' to quit): ")
    if user_text.strip().lower() == "exit":
        break

    content = [{"type": "text", "text": user_text}]

    upload_file = input("Upload a file? (y/n): ").strip().lower() == 'y'
    if upload_file:
        file_path = input("Please enter file path: ").strip()
        if os.path.exists(file_path):
            file_content = build_file_content(file_path)
            if file_content:
                content.append(file_content)
        else:
            print("File path does not exist, skipping file upload.")

    messages.append({"role": "user", "content": content})

    chat_response = client.chat.completions.create(
        model="<model_path>",
        messages=messages,
        extra_body={"stop_token_ids": [248044, 248046]},
    )

    ai_message = chat_response.choices[0].message
    print("MiniCPM-V 4.6:", ai_message.content)

    messages.append({"role": "assistant", "content": ai_message.content})
```

## 3. Offline Inference

```python
from transformers import AutoProcessor
from PIL import Image
from vllm import LLM, SamplingParams

MODEL_NAME = "<model_path>"
# Or directly:
# MODEL_NAME = "openbmb/MiniCPM-V-4.6"

image = Image.open("./assets/airplane.jpeg").convert("RGB")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

llm = LLM(
    model=MODEL_NAME,
    max_model_len=8192,
    trust_remote_code=True,
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

inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image,
        # for multi-image: "image": [image1, image2]
    },
}

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

1. **Model Path**: replace `<model_path>` with your local path or one of `openbmb/MiniCPM-V-4.6` / `openbmb/MiniCPM-V-4.6-Thinking`.
2. **Stop Tokens**: v4.6 uses the Qwen3.5 vocabulary; the correct `stop_token_ids` are `[248044, 248046]` (v4.5 used `[1, 151645]`).
3. **API Key**: ensure the key passed to `vllm serve` matches the client.
4. **Memory**: tune `--gpu_memory_utilization`, `--max-model-len`, and `--max-num-batched-tokens` for your hardware. v4.6 supports up to 256K context, but you typically don't need it.
5. **Multimodal Limits**: set `--limit-mm-per-prompt` for multi-image / multi-video sessions.
