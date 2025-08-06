# llama.cpp

```{dropdown} llama.cpp as a C++ library

Before starting, let’s first discuss what is llama.cpp and what you should expect, and why we say “use” llama.cpp, with “use” in quotes. llama.cpp is essentially a different ecosystem with a different design philosophy that targets light-weight footprint, minimal external dependency, multi-platform, and extensive, flexible hardware support:

*   Plain C/C++ implementation without external dependencies
*   Support a wide variety of hardware:
    *   AVX, AVX2 and AVX512 support for x86_64 CPU
    *   Apple Silicon via Metal and Accelerate (CPU and GPU)
    *   NVIDIA GPU (via CUDA), AMD GPU (via hipBLAS), Intel GPU (via SYCL), Ascend NPU (via CANN), and Moore Threads GPU (via MUSA)
    *   Vulkan backend for GPU
*   Various quantization schemes for faster inference and reduced memory footprint
*   CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

It’s like the Python frameworks `torch`+`transformers` or `torch`+`vllm` but in C++. However, this difference is crucial:

*   **Python is an interpreted language**: The code you write is executed line-by-line on-the-fly by an interpreter. You can run the example code snippet or script with an interpreter or a natively interactive interpreter shell. In addition, Python is learner friendly, and even if you don’t know much before, you can tweak the source code here and there.
*   **C++ is a compiled language**: The source code you write needs to be compiled beforehand, and it is translated to machine code and an executable program by a compiler. The overhead from the language side is minimal. You do have source code for example programs showcasing how to use the library. But it is not very easy to modify the source code if you are not verse in C++ or C.

To use llama.cpp means that you use the llama.cpp library in your own program, like writing the source code of [Ollama](https://ollama.com/), [GPT4ALL](https://gpt4all.io/), [llamafile](https://github.com/Mozilla-Ocho/llamafile) etc. But that’s not what this guide is intended or could do. Instead, here we introduce how to use the `llama-cli` example program, in the hope that you know that llama.cpp does support MiniCPM-V 4.0 and how the ecosystem of llama.cpp generally works.

```

In this guide, we will show how to "use" [llama.cpp](https://github.com/ggml-org/llama.cpp) to run models on your local machine, in particular, the `llama-cli` and the `llama-server` example program, which comes with the library.

The main steps:

1. Get the programs
2. Get the MiniCPM-V 4.0 models in GGUF[^1] format
3. Run the program with the model

## Getting the Program

You can get the programs in various ways. For optimal efficiency, we recommend compiling the programs locally, so you get the CPU optimizations for free. However, if you don’t have C++ compilers locally, you can also install using package managers or downloading pre-built binaries. They could be less efficient but for non-production example use, they are fine.

::::{tab-set}

:::{tab-item} Compile Locally
Here are the basic command to compile llama-cli locally on macOS or Linux. For Windows or GPU users, please refer to the guide from [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

### Installing Build Tools

To build locally, a C++ compiler and a build system tool are required. To see if they have been installed already, type `cc --version` or `cmake --version` in a terminal window.

- If installed, the build configuration of the tool will be printed and you are good to go!
- If errors are raised, you need to first install the related tools:
  - On macOS, install with the command `xcode-select --install`.
  - On Ubuntu, install with the command `sudo apt install build-essential`. For other Linux distributions, the command may vary; the essential packages needed for this guide are `gcc` and `cmake`.

### Compiling the Program

Clone the llama.cpp repository

    git clone https://github.com/ggml-org/llama.cpp
    cd llama.cpp

Build llama.cpp using `CMake`: [https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)

**CPU/Metal:**

    cmake -B build
    cmake --build build --config Release

**CUDA:**

    cmake -B build -DGGML_CUDA=ON
    cmake --build build --config Release

Based on your CPU cores, you can enable parallel compiling to shorten the time, for example:

    # build programs with 8 parallel compiling jobs
    cmake --build build --config Release -j 8

The built programs will be in `./build/bin/`.

:::

:::{tab-item} Package Managers
For macOS and Linux users, `llama-cli` and `llama-server` can be installed with package managers including Homebrew, Nix, and Flox.

Here, we show how to install `llama-cli` and` llama-server` with Homebrew. For other package managers, please check the instructions [here](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md).

Steps of installing with Homebrew:

First, Ensure that Homebrew is available on your operating system. If you don’t have Homebrew, install it as in its [website](https://brew.sh/).

Second, install the pre-built binaries with a single command:

    brew install llama.cpp

```{attention}
The installed binaries might not be built with the optimal compile options for your hardware, which can lead to poor performance. They also don’t support GPU on Linux systems.
```

:::

:::{tab-item} Binary Release

You can also download pre-built binaries from [GitHub Release](https://github.com/ggml-org/llama.cpp/releases). Please note that those pre-built binaries files are architecture-, backend-, and os-specific. If you are not sure what those mean, you probably don’t want to use them and running with incompatible versions will most likely fail or lead to poor performance.

The file names are like `llama-<version>-bin-<os>-<feature>-<arch>.zip`.

*   `<version>`: The version of llama.cpp. The latest is preferred, but as llama.cpp is updated and released frequently, the latest may contain bugs. If the latest version does not work, try the previous release until it works.
*   `<os>`: the operating system. `win` for Windows; `macos` for macOS; `linux` for Linux.
*   `<arch>`: the system architecture. `x64` for `x86_64`, e.g., most Intel and AMD systems, including Intel Mac; `arm64` for arm64, e.g., Apple Silicon or Snapdragon-based systems.

For Windows, the `<feature>` reference:

*   On CPU
    *   `x86_64` CPUs: try `avx2` first.
        *   `noavx`: No hardware acceleration at all.
        *   `avx2`, `avx`, `avx512`: SIMD-based acceleration. Most modern desktop CPUs should support avx2, and some CPUs support `avx512`.
        *   `openblas`: Relying on OpenBLAS for acceleration for prompt processing but not generation.
    *   `arm64` CPUs: try `llvm` first.
    ```{note}
    `llvm` and `msvc` are different compilers: [https://github.com/ggml-org/llama.cpp/pull/7191](https://github.com/ggml-org/llama.cpp/pull/7191)
    ```

*   On GPU: try the `cu<cuda_verison>` one for NVIDIA GPUs, `kompute` for AMD GPUs, and `sycl` for Intel GPUs first. Ensure the related drivers installed.
    *   `vulkan`: support certain NVIDIA and AMD GPUs
    *   `kompute`: support certain NVIDIA and AMD GPUs
    *   `sycl`: Intel GPUs, oneAPI runtime is included
    *   `cu<cuda_verison>`: NVIDIA GPUs, CUDA runtime is not included. You can download the `cudart-llama-bin-win-cu<cuda_version>-x64.zip` and unzip it to the same directory if you don’t have the corresponding CUDA toolkit installed.

For macOS or Linux:

*   Linux: only `llama-<version>-bin-linux-x64.zip`, supporting CPU.
*   macOS: `llama-<version>-bin-macos-x64.zip` for Intel Mac with no GPU support; `llama-<version>-bin-macos-arm64.zip` for Apple Silicon with GPU support.

Download and unzip the .zip file into a directory and open a terminal at that directory.

:::

::::

## Getting the GGUF

GGUF[^1] is a file format for storing information needed to run a model, including but not limited to model weights, model hyperparameters, default generation configuration, and tokenizer.

You can use our official GGUF files or prepare your own GGUF file.

### Download official MiniCPM-V 4.0 GGUF files

Download converted language model file (e.g., `ggml-model-Q4_K_M.gguf`) and vision model file (`mmproj-model-f16.gguf`) from:
*   HuggingFace: https://huggingface.co/openbmb/MiniCPM-V-4-gguf
*   ModelScope: https://modelscope.cn/models/OpenBMB/MiniCPM-V-4-gguf

Or download the GGUF model with `huggingface-cli` (install with `pip install huggingface_hub`):

```bash
huggingface-cli download <model_repo> <gguf_file> --local-dir <local_dir>
```

For example:

```bash
huggingface-cli download openbmb/MiniCPM-V-4-gguf ggml-model-Q4_K_M.gguf --local-dir .
```

This will download the MiniCPM-V 4.0 model in GGUF format quantized with the scheme Q4_K_M.

### Convert from PyTorch model

Model files from Hugging Face Hub can be converted to GGUF, using the `convert-hf-to-gguf.py` script. It does require you to have a working Python environment with at least `transformers` installed.

Download the MiniCPM-V-4 PyTorch model to "MiniCPM-V-4" folder:
*   HuggingFace: https://huggingface.co/openbmb/MiniCPM-V-4
*   ModelScope: https://modelscope.cn/models/OpenBMB/MiniCPM-V-4

Clone the llama.cpp repository:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Convert the PyTorch model to GGUF:

```bash
python ./tools/mtmd/legacy-models/minicpmv-surgery.py -m ../MiniCPM-V-4

python ./tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py -m ../MiniCPM-V-4 --minicpmv-projector ../MiniCPM-V-4/minicpmv.projector --output-dir ../MiniCPM-V-4/ --minicpmv_version 5

python ./convert_hf_to_gguf.py ../MiniCPM-V-4/model

# quantize int4 version
cd build/bin/
./llama-quantize ../MiniCPM-V-4/model/Model-3.6B-F16.gguf ../MiniCPM-V-4/model/ggml-model-Q4_K_M.gguf Q4_K_M
```

## Run MiniCPM-V 4.0 with llama.cpp

### llama-cli

[`llama-cli`](https://github.com/ggml-org/llama.cpp/tree/master/tools/main) is a console program which can be used to chat with LLMs. Simple run the following command where you place the llama.cpp programs:

```bash
cd build/bin/

# run f16 version
./llama-mtmd-cli -m ../MiniCPM-V-4/model/Model-3.6B-F16.gguf --mmproj ../MiniCPM-V-4/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"

# run quantized int4 version
./llama-mtmd-cli -m ../MiniCPM-V-4/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-4/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"

# or run in interactive mode
./llama-mtmd-cli -m ../MiniCPM-V-4/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-4/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -i
```

Simple argument reference:

| Argument | `-m, --model` | `--mmproj` | `--image` | `-p, --prompt` | `-c, --ctx-size` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Description | Path to the language model | Path to the vision model | Path to the input image | The prompt | Maximum context size |

Here are more detailed explanations to the command:

*   **Model**: `llama-cli` supports using model files from local path, Hugging Face hub, or remote URL.
    *   To use a local path, pass `-m ggml-model-Q4_K_M.gguf`
    *   To use the model file from Hugging Face hub, pass `-hf openbmb/MiniCPM-V-4-gguf:Q4_K_M`
    *   To use a remote URL, pass `-mu https://huggingface.co/openbmb/MiniCPM-V-4-gguf/resolve/main/ggml-model-Q4_K_M.gguf?download=true`.

*   **Speed Optimization**:
    *   CPU: `llama-cli` by default will use CPU and you can change `-t` to specify how many threads you would like it to use, e.g., `-t 8` means using 8 threads.
    *   GPU: If the programs are built with GPU support, you can use `-ngl`, which allows offloading some layers to the GPU for computation. If there are multiple GPUs, it will offload to all the GPUs. You can use `-dev` to control the devices used and `-sm` to control which kinds of parallelism is used. For example, `-ngl 99 -dev cuda0,cuda1 -sm row` means offload all layers to GPU 0 and GPU1 using the split mode row. Adding `-fa` may also speed up the generation.

*   **Sampling Parameters**: llama.cpp supports a variety of [sampling methods](https://github.com/ggml-org/llama.cpp/tree/master/tools/main#generation-flags) and has default configuration for many of them. It is recommended to adjust those parameters according to the actual case and the recommended parameters from MiniCPM-V 4.0 modelcard could be used as a reference. If you encounter repetition and endless generation, it is recommended to pass in addition `--presence-penalty` up to `2.0`.

*   **Context Management**: llama.cpp adopts the “rotating” context management by default. The `-c` controls the maximum context length (default 4096, 0 means loaded from model), and `-n` controls the maximum generation length each time (default -1 means infinite until ending, -2 means until context full). When the context is full but the generation doesn’t end, the first `--keep` tokens (default 0, -1 means all) from the initial prompt is kept, and the first half of the rest is discarded. Then, the model continues to generate based on the new context tokens. You can set `--no-context-shift` to prevent this rotating behavior and the generation will stop once `-c` is reached. llama.cpp supports YaRN, which can be enabled by `-c 131072 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`.

*   **Chat**: `--jinja` indicates using the chat template embedded in the GGUF which is preferred and `--color` indicates coloring the texts so that user input and model output can be better differentiated. If there is a chat template, `llama-cli` will enter chat mode automatically. To stop generation or exit press "Ctrl+C". You can use `-sys` to add a system prompt.

### llama-server

[llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) is a simple HTTP server, including a set of LLM REST APIs and a simple web front end to interact with LLMs using llama.cpp.

he core command is similar to that of llama-cli. In addition, it supports thinking content parsing and tool call parsing.

```bash
./llama-server -m ../MiniCPM-V-4/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-V-4/mmproj-model-f16.gguf -ngl 100 -...
```

By default, the server will listen at `http://localhost:8080` which can be changed by passing `--host` and `--port`. The web front end can be assessed from a browser at `http://localhost:8080/`. The OpenAI compatible API is at `http://localhost:8080/v1/`.

## What’s More

If you still find it difficult to use llama.cpp, don’t worry, just check out other llama.cpp-based applications. For example, MiniCPM-V 4.0 has already been officially part of [Ollama](https://ollama.com/), which is a good platform for you to search and run local LLMs.

Have fun!

[^1]: GGUF (GPT-Generated Unified Format) is a file format designed for efficiently storing and loading language models for inference.
