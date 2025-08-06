# iOS Demo

## 1. Deploying iOS App

**NOTE: To deploy and test the app on an iOS device, you may need an Apple Developer account.**

Clone the official iOS demo repository provided by OpenBMB:

```bash
git clone https://github.com/tc-mb/MiniCPM-o-demo-iOS.git
cd MiniCPM-o-demo-iOS
```

Install Xcode:

* Download Xcode from the App Store
* Install the Command Line Tools:

  ```bash
  xcode-select --install
  ```
* Agree to the software license agreement:

  ```bash
  sudo xcodebuild -license
  ```

Open `MiniCPM-demo.xcodeproj` with Xcode. It may take a moment for Xcode to automatically download the required dependencies.

In Xcode, select the target device at the top of the window, then click the "Run" (triangle) button to launch the demo.

**NOTE: If you encounter errors related to the `thirdparty/llama.xcframework` path, please follow the steps below to build the `llama.xcframework` manually.**

## 2. Manually Building the llama.cpp Library From OpenBMB

Clone the llama.cpp repository:

```bash
git clone https://github.com/tc-mb/llama.cpp.git
cd llama.cpp
```

Build the llama.cpp library for iOS using the script:

```bash
./build-xcframework.sh
```

Copy the built library into the corresponding directory of the iOS demo project:

```bash
cp -r ./build-apple/llama.xcframework ../MiniCPM_demo_iOS/MiniCPM-demo/thirdparty
```

## 3. GGUF Files

### 1: Download Official GGUF Files

* HuggingFace: [https://huggingface.co/openbmb/MiniCPM-V-4-gguf](https://huggingface.co/openbmb/MiniCPM-V-4-gguf)
* ModelScope: [https://modelscope.cn/models/OpenBMB/MiniCPM-V-4-gguf](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4-gguf)

Download the language model file (e.g., `ggml-model-Q4_K_M.gguf`) and the vision model file (`mmproj-model-f16.gguf`) from the repository.

### 2: Convert from PyTorch Model

Download the MiniCPM-V-4 PyTorch model into a folder named `MiniCPM-V-4`:

* HuggingFace: [https://huggingface.co/openbmb/MiniCPM-V-4](https://huggingface.co/openbmb/MiniCPM-V-4)
* ModelScope: [https://modelscope.cn/models/OpenBMB/MiniCPM-V-4](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4)

Convert the PyTorch model to GGUF format:

```bash
python ./tools/mtmd/legacy-models/minicpmv-surgery.py -m ../MiniCPM-V-4

python ./tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py -m ../MiniCPM-V-4 --minicpmv-projector ../MiniCPM-V-4/minicpmv.projector --output-dir ../MiniCPM-V-4/ --minicpmv_version 5

python ./convert_hf_to_gguf.py ../MiniCPM-V-4/model

# int4 quantized
./llama-quantize ../MiniCPM-V-4/model/Model-3.6B-F16.gguf ../MiniCPM-V-4/model/ggml-model-Q4_K_M.gguf Q4_K_M
```
