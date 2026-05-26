# MiniCPM-V 4.6 vLLM 部署指南

MiniCPM-V 4.6 提供 **两个独立的 checkpoint**：

| 版本          | HuggingFace ID                                                                                   | 魔搭社区 ID                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Instruct      | [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6)                            | [OpenBMB/MiniCPM-V-4.6](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6)                       |
| Thinking（思考） | [openbmb/MiniCPM-V-4.6-Thinking](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking)                | [OpenBMB/MiniCPM-V-4.6-Thinking](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-Thinking)           |

> 与 v4.5 通过 `enable_thinking` 切换模式不同，v4.6 将 instruct 与 thinking 拆分为 **两个独立 checkpoint**，按需选择即可。

## 1. 环境准备

### 1.1 安装 vLLM

MiniCPM-V 4.6 自 [PR #43213](https://github.com/vllm-project/vllm/pull/43213)（2026-05-22 merged）起在官方 vLLM `main` 分支上完整可用。**无需使用 fork**，直接从官方仓库安装即可，确保检出的 commit ≥ 该 PR 的 merge 时间：

```bash
# 新建干净的 conda 环境
conda create -n vllm-v46 python=3.10 -y
conda activate vllm-v46

# 从官方 main 源码安装（需要 CUDA 工具链；HEAD 需包含 #43213）
git clone https://github.com/vllm-project/vllm.git
cd vllm
MAX_JOBS=6 VLLM_USE_PRECOMPILED=1 pip install --editable . -v
```

> [!TIP]
> 等到 vLLM 发布包含 #43213 的正式 release 之后，可直接 `pip install -U vllm` 升级，无需源码安装。请关注 [vLLM Releases](https://github.com/vllm-project/vllm/releases)。

进行视频推理时，需要安装相应的视频模块：

```bash
pip install vllm[video]
```

vLLM 要求 `transformers>=5.7.0`，MiniCPM-V 4.6 在 transformers 中以独立架构 `MiniCPMV4_6ForConditionalGeneration` 形式合并。

校验安装：

```bash
python -c "import vllm, transformers; print('vllm', vllm.__version__, '| transformers', transformers.__version__)"
```

## 2. API 服务部署

### 2.1 启动 API 服务

```bash
vllm serve <模型路径> \
  --dtype auto \
  --max-model-len 8192 \
  --api-key token-abc123 \
  --gpu_memory_utilization 0.9 \
  --trust-remote-code \
  --max-num-batched-tokens 8192
```

**参数说明：**
- `<模型路径>`：MiniCPM-V-4.6 的本地路径，或 HuggingFace ID `openbmb/MiniCPM-V-4.6` / `openbmb/MiniCPM-V-4.6-Thinking`
- `--api-key`：API 访问密钥
- `--max-model-len`：最大上下文长度。v4.6 backbone 最长支持 256K，部署时按需设置即可
- `--gpu_memory_utilization`：GPU 显存使用率

### 2.2 图片推理

```python
from openai import OpenAI
import base64

openai_api_key = "token-abc123"  # 需与启动服务时设置的密钥保持一致
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

with open('./assets/airplane.jpeg', 'rb') as file:
    image = "data:image/jpeg;base64," + base64.b64encode(file.read()).decode('utf-8')

chat_response = client.chat.completions.create(
    model="<模型路径>",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "请描述这张图片"},
            {"type": "image_url", "image_url": {"url": image}},
        ],
    }],
    extra_body={
        # v4.6 使用 Qwen3.5 词表，stop_token_ids 与 v4.5 不同
        "stop_token_ids": [248044, 248046]
    }
)

print("Chat response:", chat_response)
print("Chat response content:", chat_response.choices[0].message.content)
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
    model="<模型路径>",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这个视频"},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_base64}"}},
            ],
        },
    ],
    extra_body={"stop_token_ids": [248044, 248046]}
)

print(chat_response.choices[0].message.content)
```

> v4.6 的视频通路非常简单：图像处理器逐帧调用，单视频帧切片预算由 `mm_processor_kwargs` / `video_processor.max_slice_nums` 控制，**无需** `enhanced_opencv` 等特殊后端。

调整单视频帧切片数量：

```python
extra_body = {
    "stop_token_ids": [248044, 248046],
    "mm_processor_kwargs": {"max_slice_nums": 2},
}
```

### 2.4 思考模式

如果部署的是 **`openbmb/MiniCPM-V-4.6-Thinking`** 模型，chat template 会默认在 assistant 起始位置插入 `<think>` 块；模型先输出推理过程，再以 `</think>` 分隔后给出回答。可通过 `chat_template_kwargs` 跳过：

```python
extra_body = {
    "stop_token_ids": [248044, 248046],
    # 在 Thinking 模型上跳过开头的 <think> 块
    "chat_template_kwargs": {"enable_thinking": False},
}
```

Instruct 模型不会输出 `<think>` 块，按业务需求选择对应 checkpoint 即可，**不再像 v4.5 那样需要在请求里切换模式**。

### 2.5 多轮对话

#### 启动参数配置

进行多图 / 多视频对话时，需在启动时调大 `--limit-mm-per-prompt`：

```bash
# 视频多轮对话（最多 3 个视频）
vllm serve <模型路径> --dtype auto --max-model-len 16384 --api-key token-abc123 \
  --gpu_memory_utilization 0.9 --trust-remote-code \
  --limit-mm-per-prompt '{"video": 3}'
```

```bash
# 图片和视频混合输入
vllm serve <模型路径> --dtype auto --max-model-len 16384 --api-key token-abc123 \
  --gpu_memory_utilization 0.9 --trust-remote-code \
  --limit-mm-per-prompt '{"image": 5, "video": 2}'
```

#### 多轮对话示例代码

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
        print(f"不支持的文件类型: {mime_type}")
        return None

while True:
    user_text = input("请输入问题（输入 'exit' 退出）：")
    if user_text.strip().lower() == "exit":
        break

    content = [{"type": "text", "text": user_text}]

    upload_file = input("是否上传文件？(y/n): ").strip().lower() == 'y'
    if upload_file:
        file_path = input("请输入文件路径: ").strip()
        if os.path.exists(file_path):
            file_content = build_file_content(file_path)
            if file_content:
                content.append(file_content)
        else:
            print("文件路径不存在，跳过文件上传。")

    messages.append({"role": "user", "content": content})

    chat_response = client.chat.completions.create(
        model="<模型路径>",
        messages=messages,
        extra_body={"stop_token_ids": [248044, 248046]},
    )

    ai_message = chat_response.choices[0].message
    print("MiniCPM-V 4.6:", ai_message.content)

    messages.append({"role": "assistant", "content": ai_message.content})
```

## 3. 离线推理

```python
from transformers import AutoProcessor
from PIL import Image
from vllm import LLM, SamplingParams

MODEL_NAME = "<模型路径>"
# 或直接使用 HuggingFace ID：
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
        {"type": "text", "text": "请描述这张图片的内容"},
    ],
}]

prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image,
        # 多图推理：
        # "image": [image1, image2]
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

## 注意事项

1. **模型路径**：将示例中的 `<模型路径>` 替换为本地路径，或 `openbmb/MiniCPM-V-4.6` / `openbmb/MiniCPM-V-4.6-Thinking`。
2. **停止符**：v4.6 采用 Qwen3.5 词表，正确的 `stop_token_ids` 为 `[248044, 248046]`（v4.5 用的是 `[1, 151645]`）。
3. **API 密钥**：客户端密钥需与 `vllm serve` 启动时一致。
4. **显存配置**：根据硬件调整 `--gpu_memory_utilization` / `--max-model-len` / `--max-num-batched-tokens`。v4.6 backbone 最长支持 256K，但通常无需开满。
5. **多模态限制**：多图 / 多视频对话时设置合适的 `--limit-mm-per-prompt`。
