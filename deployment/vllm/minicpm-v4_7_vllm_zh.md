# MiniCPM-V 4.7 — vLLM 部署指南

MiniCPM-V 4.7 提供**两个尺寸**，vLLM 均已原生支持：

| 尺寸              | HuggingFace ID                                                                                   | ModelScope ID                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| 1B（dense）       | [openbmb/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)                       | [OpenBMB/MiniCPM-V-4.7-1B](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B)                  |
| 35B-A3B（MoE）    | [openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)             | [OpenBMB/MiniCPM-V-4.7-35B-A3B](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B)        |

> MiniCPM-V 4.7 注册为 `MiniCPMV4_7ForConditionalGeneration`，已在
> [PR #58674](https://github.com/vllm-project/vllm/pull/58674) 合入官方 vLLM `main`。
> **无需 fork** —— 从上游安装即可，确保 checkout 在该提交之后。
> vLLM 需要 **`transformers >= 5.18.0`**（4.7 在其中为原生架构）。

## 1. 环境准备

```bash
# 新建干净的 conda 环境
conda create -n vllm-v47 python=3.10 -y
conda activate vllm-v47

# 从上游 main 安装（需 CUDA toolkit；HEAD 需含 #58674）
git clone https://github.com/vllm-project/vllm.git
cd vllm
MAX_JOBS=6 VLLM_USE_PRECOMPILED=1 pip install --editable . -v
```

> [!TIP]
> 一旦含 #58674 的 vLLM 正式版发布，`pip install -U vllm` 即可，无需编译。

视频推理需安装视频模块：

```bash
pip install vllm[video]
```

验证安装：

```bash
python -c "import vllm, transformers; print('vllm', vllm.__version__, '| transformers', transformers.__version__)"
```

## 2. API 服务部署

### 2.1 启动 API 服务

```bash
vllm serve <model_path> \
  --dtype auto \
  --max-model-len 8192 \
  --api-key token-abc123 \
  --gpu_memory_utilization 0.9 \
  --max-num-batched-tokens 8192
```

- `<model_path>`：本地路径，或 `openbmb/MiniCPM-V-4.7-1B` / `openbmb/MiniCPM-V-4.7-35B-A3B`。
  **无需 `--trust-remote-code`** —— 架构在 `transformers >= 5.18.0` 中原生支持。
- `--api-key`：API 访问密钥。
- `--max-model-len`：最大上下文长度（4.7 支持到 256K，按显存先设小）。
- `--gpu_memory_utilization`：显存利用率。

### 2.2 图片推理

```python
from openai import OpenAI
import base64

openai_api_key = "token-abc123"  # 需与 vllm serve 传入的一致
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

with open('./assets/airplane.jpeg', 'rb') as file:
    image = "data:image/jpeg;base64," + base64.b64encode(file.read()).decode('utf-8')

chat_response = client.chat.completions.create(
    model="<model_path>",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "请描述这张图片"},
            {"type": "image_url", "image_url": {"url": image}},
        ],
    }],
    extra_body={
        # 4.7 使用 Qwen3.5 词表，stop token 与 4.6 相同
        "stop_token_ids": [248044, 248046],
        # 关闭 <think> 块以获得直答
        "chat_template_kwargs": {"enable_thinking": False},
    }
)

print(chat_response.choices[0].message.content)
```

### 2.3 视频推理

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
                {"type": "text", "text": "请描述这段视频"},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_base64}"}},
            ],
        },
    ],
    extra_body={"stop_token_ids": [248044, 248046]},
)
print(chat_response.choices[0].message.content)
```

调整单视频帧切片预算：

```python
extra_body = {
    "stop_token_ids": [248044, 248046],
    "mm_processor_kwargs": {"max_slice_nums": 2},
}
```

### 2.4 思考模式

4.7 每个尺寸只有一个 checkpoint：对话模板默认开启 `<think>`。可按请求关闭：

```python
extra_body = {
    "stop_token_ids": [248044, 248046],
    "chat_template_kwargs": {"enable_thinking": False},
}
```

### 2.5 多图 / 多视频

启动时提高 `--limit-mm-per-prompt` 预算：

```bash
# 单请求最多 5 图 + 2 视频
vllm serve <model_path> --dtype auto --max-model-len 16384 --api-key token-abc123 \
  --gpu_memory_utilization 0.9 \
  --limit-mm-per-prompt '{"image": 5, "video": 2}'
```

## 3. 离线推理

```python
from transformers import AutoProcessor
from PIL import Image
from vllm import LLM, SamplingParams

MODEL_NAME = "<model_path>"  # 或 "openbmb/MiniCPM-V-4.7-1B"

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
        {"type": "text", "text": "请描述这张图片的内容"},
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

## 说明

1. **模型路径**：`<model_path>` 换成本地路径或 `openbmb/MiniCPM-V-4.7-1B` / `openbmb/MiniCPM-V-4.7-35B-A3B`。
2. **Stop tokens**：`[248044, 248046]`（Qwen3.5 词表；v4.5 是 `[1, 151645]`）。
3. **显存**：1B 单张消费级卡即可；**35B-A3B**（bf16 约 70 GB）需多卡或 4-bit 权重
   （`-AWQ` / `-GPTQ` / `-FP8` / `-INT4`）。按硬件调 `--gpu_memory_utilization` / `--max-model-len` / `--max-num-batched-tokens`。
4. **多模态上限**：多图/多视频会话用 `--limit-mm-per-prompt`。
5. **无需 `trust_remote_code`**：4.7 在 `transformers >= 5.18.0` 中原生支持。

## 推荐采样参数

在 instruct 模式下生成时，推荐使用：`temperature=0.7`、`top_p=0.95`、`top_k=100`、`min_p=0.0`、`presence_penalty=0`、`repetition_penalty=1.03`。
