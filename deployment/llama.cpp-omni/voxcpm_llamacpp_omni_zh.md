# VoxCPM - llama.cpp-omni

> [!NOTE]
> VoxCPM 是面壁智能推出的语音合成（TTS）模型，支持声音设计、声音克隆和流式输出。其 C++/GGML 推理由 [llama.cpp-omni](https://github.com/OpenBMB/llama.cpp-omni) 项目提供。
>
> 目前支持三个版本：**VoxCPM-0.5B**、**VoxCPM-1.5** 和 **VoxCPM2**，支持 CPU 和 CUDA 后端。

## 支持的模型

三个版本共享相同的推理流程：

```
BaseLM -> ResidualLM -> FSQ -> LocEnc/LocDiT CFM -> AudioVAE
```

| 模型 | BaseLM | ResidualLM | FSQ 维度 | AudioVAE 输出 |
|------|--------|------------|---------|---------------|
| VoxCPM-0.5B | 24 层, h=1024 | 6 层, h=1024 | 256 | 16 kHz |
| VoxCPM-1.5 | 24 层, h=1024 | 8 层, h=1024 | 256 | 44.1 kHz |
| VoxCPM2 | 28 层, h=2048 | 8 层, h=2048 | 512 | 48 kHz |

## 1. 克隆并编译

克隆 llama.cpp-omni 仓库：

```bash
git clone https://github.com/OpenBMB/llama.cpp-omni.git
cd llama.cpp-omni
```

使用 CMake 构建（CPU 或 CUDA）：

**仅 CPU：**

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target voxcpm2-cli
```

**CUDA：**

```bash
cmake -B build -S . -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target voxcpm2-cli
```

输出二进制文件：`build/bin/voxcpm2-cli`

## 2. 获取模型权重

### 方法一：下载已转换的 GGUF 文件

从以下平台下载对应版本的 BaseLM 和 Acoustic GGUF 文件：

- HuggingFace：<https://huggingface.co/openbmb>
- 魔搭社区：<https://modelscope.cn/models/OpenBMB>

### 方法二：从 PyTorch 模型转换

安装转换依赖：

```bash
pip install torch safetensors numpy gguf
```

将 PyTorch 权重转换为 GGUF 格式：

```bash
cd tools/omni/voxcpm2

python convert_voxcpm2_to_gguf.py \
    --model /path/to/model.safetensors_or_pytorch_model.bin \
    --vae /path/to/audiovae.pth \
    --config /path/to/config.json \
    --output /path/to/output
```

默认输出 F16 GGUF 文件。如需生成 F32 文件（用于量化）：

```bash
python convert_voxcpm2_to_gguf.py \
    --model /path/to/model.safetensors \
    --vae /path/to/audiovae.pth \
    --config /path/to/config.json \
    --output /path/to/output \
    --dtype f32
```

默认输出文件名：

| 模型 | BaseLM | Acoustic |
|------|--------|----------|
| VoxCPM-0.5B | `VoxCPM-0.5B-BaseLM-F16.gguf` | `VoxCPM-0.5B-Acoustic-F16.gguf` |
| VoxCPM-1.5 | `VoxCPM-1.5-BaseLM-F16.gguf` | `VoxCPM-1.5-Acoustic-F16.gguf` |
| VoxCPM2 | `VoxCPM2-BaseLM-F16.gguf` | `VoxCPM2-Acoustic-F16.gguf` |

### BaseLM 量化（Q8_0）

BaseLM GGUF 可量化为 Q8_0，模型体积缩小约 2 倍，质量损失极小。需要 F32 输入：

```bash
# 1. 转换为 F32
python convert_voxcpm2_to_gguf.py \
    --model /path/to/model.safetensors \
    --vae /path/to/audiovae.pth \
    --config /path/to/config.json \
    --output ./gguf \
    --dtype f32

# 2. 编译 llama-quantize
cmake --build build -j$(nproc) --target llama-quantize

# 3. 将 BaseLM 量化为 Q8_0
./build/bin/llama-quantize \
    ./gguf/VoxCPM2-BaseLM-F32.gguf \
    ./gguf/VoxCPM2-BaseLM-Q8_0.gguf \
    Q8_0
```

## 3. CLI 推理

### 基础 TTS

```bash
./build/bin/voxcpm2-cli \
    -t "你好，欢迎使用 VoxCPM2。" \
    -o output.wav \
    VoxCPM2-BaseLM-F16.gguf \
    VoxCPM2-Acoustic-F16.gguf
```

### 声音设计

在文本开头用括号描述期望的声音特征：

```bash
./build/bin/voxcpm2-cli \
    -t "(一位年轻女性，温柔甜美的声音)你好，欢迎使用 VoxCPM2！" \
    -o voice_design.wav \
    VoxCPM2-BaseLM-F16.gguf \
    VoxCPM2-Acoustic-F16.gguf
```

### 声音克隆

提供参考音频文件（WAV 格式，单声道，任意采样率）：

```bash
./build/bin/voxcpm2-cli \
    -t "这是由 VoxCPM2 生成的克隆声音。" \
    -r speaker.wav \
    -o clone.wav \
    VoxCPM2-BaseLM-F16.gguf \
    VoxCPM2-Acoustic-F16.gguf
```

### 流式输出

```bash
./build/bin/voxcpm2-cli \
    -t "VoxCPM2 流式输出测试。" \
    --stream \
    -o streaming.wav \
    VoxCPM2-BaseLM-F16.gguf \
    VoxCPM2-Acoustic-F16.gguf
```

### CLI 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-t, --text` | （必填） | 输入文本 |
| `-o, --output` | `output.wav` | 输出 WAV 文件路径 |
| `-r, --reference` | — | 用于声音克隆的参考 WAV 文件 |
| `--stream` | — | 流式输出模式 |
| `--steps` | 200 | 最大解码步数（非流式纯文本 TTS 时会根据文本长度自动缩减） |
| `--timesteps` | 10 | CFM 推理时间步数 |
| `--cfg` | 2.0 | CFG 引导强度 |
| `--temperature` | 1.0 | 噪声温度 |
| `--seed` | 42 | 随机种子 |
| `--cpu` | — | 强制使用 CPU 后端（默认：GPU） |
| `--n-gpu-layers` | -1 | 卸载到 GPU 的层数（-1 = 全部） |

## 4. 在线服务（OpenAI 兼容 API）

`llama-server` 二进制文件在 `/v1/audio/speech` 路径提供 OpenAI 兼容的 TTS 端点。

### 编译服务端

```bash
cmake --build build -j$(nproc) --target llama-server
```

### 启动服务

**VoxCPM2 独立模式：**

```bash
./build/bin/llama-server \
    --voxcpm2-base-lm /path/to/VoxCPM2-BaseLM-F16.gguf \
    --voxcpm2-acoustic /path/to/VoxCPM2-Acoustic-F16.gguf \
    --port 8080
```

**与主 LLM 一起启动：**

```bash
./build/bin/llama-server \
    -m /path/to/llm.gguf \
    --voxcpm2-base-lm /path/to/VoxCPM2-BaseLM-F16.gguf \
    --voxcpm2-acoustic /path/to/VoxCPM2-Acoustic-F16.gguf \
    --port 8080
```

### 动态加载（无需重启）

```bash
curl http://localhost:8080/v1/voxcpm2/init \
  -H "Content-Type: application/json" \
  -d '{
    "base_lm": "/path/to/VoxCPM2-BaseLM-F16.gguf",
    "acoustic": "/path/to/VoxCPM2-Acoustic-F16.gguf",
    "n_gpu_layers": -1
  }'
```

### API 调用示例

**基础 TTS：**

```bash
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm",
    "input": "你好，欢迎使用 VoxCPM2！",
    "voice": "default"
  }' \
  --output out.wav
```

**声音设计：**

```bash
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm",
    "input": "(一位年轻女性，温柔甜美的声音)你好，欢迎使用 VoxCPM2！",
    "voice": "default"
  }' \
  --output voice_design.wav
```

**声音克隆（base64 编码的参考 WAV）：**

```bash
REF_B64=$(base64 -w0 speaker.wav)
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"voxcpm2\",
    \"input\": \"这是克隆的声音。\",
    \"voice\": \"default\",
    \"reference_audio\": \"$REF_B64\"
  }" \
  --output clone.wav
```

**流式输出：**

```bash
curl http://localhost:8080/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm",
    "input": "VoxCPM2 流式输出测试。",
    "voice": "default"
  }' \
  --output streaming.wav
```

### API 接口说明

#### `POST /v1/audio/speech`

OpenAI 兼容的 TTS 端点。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | — | 模型标识符。使用 `"voxcpm"`（支持 VoxCPM-0.5B、VoxCPM-1.5 和 VoxCPM2） |
| `input` | string | （必填） | 要合成的文本 |
| `voice` | string | `"default"` | 声音标识符（保留字段） |
| `response_format` | string | `"wav"` | 输出格式：`wav` 或 `pcm` |
| `reference_audio` | string | — | 用于声音克隆的 base64 编码 WAV |
| `seed` | int | `42` | 随机种子 |
| `cfg_value` | float | `2.0` | CFG 引导强度 |
| `inference_timesteps` | int | `10` | CFM 推理时间步数 |
| `max_steps` | int | `200` | 最大解码步数 |
| `temperature` | float | `1.0` | 噪声温度 |

#### `POST /v1/audio/speech/stream`

流式变体。以分块方式返回正在生成的 WAV 音频。

#### `POST /v1/voxcpm2/init`

动态加载或重新加载 VoxCPM2 运行时。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_lm` | string | （必填） | BaseLM GGUF 路径 |
| `acoustic` | string | （必填） | Acoustic GGUF 路径 |
| `n_gpu_layers` | int | `-1` | GPU 层数（-1 = 全部） |

#### `GET /v1/audio/speech/models`

列出已加载的 VoxCPM2 模型信息。
