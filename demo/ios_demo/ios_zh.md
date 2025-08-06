# MiniCPM-V 4.0 - iOS 设备部署

## 1. 部署 iOS App

**NOTE: 为了获取在 iOS 设备上部署、测试 App 的权限，您可能需要拥有一个 Apple 开发者账号**

克隆 iOS demo（基于`llama.cpp`）代码仓库: 

```bash
git clone https://github.com/tc-mb/MiniCPM-o-demo-iOS.git
cd MiniCPM-o-demo-iOS
```

安装 Xcode:

- 从 App Store 下载 Xcode
- 安装命令行工具
    ```bash
    xcode-select --install
- 同意软件许可协议
    ```bash
    sudo xcodebuild -license
    ```

使用 Xcode 打开 `MiniCPM-demo.xcodeproj`，可能需要等待 Xcode 自动下载所需的库。

在 Xcode 页面顶部选择想要运行 iOS demo 的设备，点击三角形的 Run 按钮即可运行。

**NOTE: 若上述流程出现 `thirdparty/llama.xcframework `路径相关报错，请按照如下教程自行编译 `llama.xcframework` 库**

## 2. 自行编译构建 OpenBMB 提供的 llama.cpp

克隆 llama.cpp 代码仓库: 
```bash
git clone https://github.com/tc-mb/llama.cpp.git
cd llama.cpp
```

使用脚本为 iOS 设备构建所需的 llama.cpp 库: 

```bash
./build-xcframework.sh
```

将构建完成的库复制到 iOS demo 对应目录:

```bash
cp -r ./build-apple/llama.xcframework ../MiniCPM_demo_iOS/MiniCPM-demo/thirdparty
```

## 3. 获取模型 GGUF 权重

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
