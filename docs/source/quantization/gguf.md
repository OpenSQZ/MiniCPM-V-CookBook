# GGUF

:::{Note}
**Support:** MiniCPM-V 4.0 / MiniCPM-V 2.6 / MiniCPM-V 2.5
:::

### 1.Download the PyTorch Model

First, obtain the original PyTorch model files from one of the following sources:

*   **HuggingFace:** https://huggingface.co/openbmb/MiniCPM-V-4
*   **ModelScope Community:** https://modelscope.cn/models/OpenBMB/MiniCPM-V-4

### 2.Convert the PyTorch Model to GGUF Format

Run the following commands in sequence to perform model surgery, convert the vision encoder, and then convert the language model.

```bash
# Step 1: Pre-process the model structure
python ./tools/mtmd/legacy-models/minicpmv-surgery.py -m ../MiniCPM-V-4

# Step 2: Convert the vision encoder to GGUF format
python ./tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py -m ../MiniCPM-V-4 --minicpmv-projector ../MiniCPM-V-4/minicpmv.projector --output-dir ../MiniCPM-V-4/ --minicpmv_version 5

# Step 3: Convert the language model to GGUF format
python ./convert_hf_to_gguf.py ../MiniCPM-V-4/model
```

### 3.Perform INT4 Quantization

Once the conversion is complete, use the `llama-quantize` tool to quantize the F16 precision GGUF model to INT4.

```bash
cd build/bin/
./llama-quantize ../MiniCPM-V-4/model/Model-3.6B-F16.gguf ../MiniCPM-V-4/model/ggml-model-Q4_K_M.gguf Q4_K_M
```
