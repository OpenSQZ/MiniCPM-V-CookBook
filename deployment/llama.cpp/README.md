# llama.cpp

## llama.cpp as a C++ library

Before starting, let's first discuss what is llama.cpp and what you should expect, and why we say "use" llama.cpp, with "use" in quotes. llama.cpp is essentially a different ecosystem with a different design philosophy that targets light-weight footprint, minimal external dependency, multi-platform, and extensive, flexible hardware support:

- Plain C/C++ implementation without external dependencies
- Support a wide variety of hardware:
  - AVX, AVX2 and AVX512 support for x86_64 CPU
  - Apple Silicon via Metal and Accelerate (CPU and GPU)
  - NVIDIA GPU (via CUDA), AMD GPU (via hipBLAS), Intel GPU (via SYCL), Ascend NPU (via CANN), and Moore Threads GPU (via MUSA)
  - Vulkan backend for GPU
- Various quantization schemes for faster inference and reduced memory footprint
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

It's like the Python frameworks `torch`+`transformers` or `torch`+`vllm` but in C++. However, this difference is crucial:

- Python is an interpreted language: The code you write is executed line-by-line on-the-fly by an interpreter. You can run the example code snippet or script with an interpreter or a natively interactive interpreter shell. In addition, Python is learner friendly, and even if you don't know much before, you can tweak the source code here and there.
- C++ is a compiled language: The source code you write needs to be compiled beforehand, and it is translated to machine code and an executable program by a compiler. The overhead from the language side is minimal. You do have source code for example programs showcasing how to use the library. But it is not very easy to modify the source code if you are not verse in C++ or C.

To use llama.cpp means that you use the llama.cpp library in your own program, like writing the source code of [Ollama](https://ollama.com/), [GPT4ALL](https://gpt4all.io/), [llamafile](https://github.com/Mozilla-Ocho/llamafile) etc. But that's not what this guide is intended or could do. Instead, here we introduce how to use the `llama-cli` example program, in the hope that you know that llama.cpp does support MiniCPM-V and how the ecosystem of llama.cpp generally works.

In this guide, we will show how to "use" [llama.cpp](https://github.com/ggml-org/llama.cpp) to run models on your local machine, in particular, the `llama-cli` and the `llama-server` example program, which comes with the library.

### The main steps

1. Get the programs
2. Get the MiniCPM-V models in GGUF[^1] format
3. Run the program with the model

## Versioned Deployment Guides

| Version | English | 中文 |
| :--- | :---: | :---: |
| **MiniCPM-V 4.7** *(latest)* | [Guide](./minicpm-v4_7_llamacpp.md) | [指南](./minicpm-v4_7_llamacpp_zh.md) |
| MiniCPM-V 4.6 | [Guide](./minicpm-v4_6_llamacpp.md) | [指南](./minicpm-v4_6_llamacpp_zh.md) |
| MiniCPM-V 4.5 | [Guide](./minicpm-v4_5_llamacpp.md) | [指南](./minicpm-v4_5_llamacpp_zh.md) |
| MiniCPM-V 4.0 | [Guide](./minicpm-v4_llamacpp.md)   | [指南](./minicpm-v4_llamacpp_zh.md)   |
| MiniCPM-o 4.5 | [Guide](./minicpm-o4_5_llamacpp.md) | [指南](./minicpm-o4_5_llamacpp_zh.md) |

[^1]: GGUF (GPT-Generated Unified Format) is a file format designed for efficiently storing and loading language models for inference.
