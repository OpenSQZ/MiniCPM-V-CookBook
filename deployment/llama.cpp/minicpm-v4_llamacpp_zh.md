# MiniCPM-V 4.0 - llama.cpp

## 1. 编译安装 llama.cpp

克隆 llama.cpp 代码仓库: 
```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

使用 `CMake` 构建 llama.cpp: 
https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md

**CPU/Metal:**
```bash
cmake -B build
cmake --build build --config Release
```

**CUDA:**
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```
## 2. 获取模型 GGUF 权重

### 方法一: 下载官方 GGUF 文件

*   HuggingFace: https://huggingface.co/openbmb/MiniCPM-V-4-gguf
*   魔搭社区: https://modelscope.cn/models/OpenBMB/MiniCPM-V-4-gguf

从仓库中下载语言模型文件（如: `ggml-model-Q4_K_M.gguf`）与视觉模型文件（`mmproj-model-f16.gguf`）

### 方法二: 从 Pytorch 模型转换

下载 MiniCPM-V-4 PyTorch 模型到 "MiniCPM-V-4" 文件夹:
*   HuggingFace: https://huggingface.co/openbmb/MiniCPM-V-4
*   魔搭社区: https://modelscope.cn/models/OpenBMB/MiniCPM-V-4

将 PyTorch 模型转换为 GGUF 格式:

```bash
python ./tools/mtmd/legacy-models/minicpmv-surgery.py -m ../MiniCPM-V-4

python ./tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py -m ../MiniCPM-V-4 --minicpmv-projector ../MiniCPM-V-4/minicpmv.projector --output-dir ../MiniCPM-V-4/ --minicpmv_version 5

python ./convert_hf_to_gguf.py ../MiniCPM-V-4/model

# int4 量化版本
./llama-quantize ../MiniCPM-V-4/model/Model-3.6B-F16.gguf ../MiniCPM-V-4/model/ggml-model-Q4_K_M.gguf Q4_K_M
```

## 3. 模型推理

```bash
cd build/bin/

# 运行 f16 版本
./llama-mtmd-cli -m ../MiniCPM-V-4/model/Model-3.6B-F16.gguf --mmproj ../MiniCPM-V-4/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"

# 运行 int4 量化版本
./llama-mtmd-cli -m ../MiniCPM-V-4/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-4/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"

# 或以交互模式运行
./llama-mtmd-cli -m ../MiniCPM-V-4/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-4/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -i
```

**命令行参数解析:**

| 参数 | `-m, --model` | `--mmproj` | `--image` | `-p, --prompt` | `-c, --ctx-size` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 含义 | 语言模型路径 | 视觉模型路径 | 输入图片路径 | 提示词 | 输入上下文最大长度 |
