# Model Deployment Guide

Multiple deployment solutions for efficient MiniCPM-o model deployment across different environments.

📖 [中文版本](./README_zh.md) | [Back to Main](../)

## Deployment Framework Comparison

| Framework | Performance | Ease of Use | Scalability | Hardware | Best For |
|-----------|-------------|-------------|-------------|----------|----------|
| [**vLLM**](./vllm/) | High | Medium | High | GPU | Large-scale production services |
| [**SGLang**](./sglang/) | High | Medium | High | GPU | Structured generation tasks |
| [**Ollama**](./ollama/) | Medium | Excellent | Medium | CPU/GPU | Personal use, rapid prototyping |
| [**Llama.cpp**](./llama.cpp/) | Medium | High | Medium | CPU | Edge devices, lightweight deployment |
| [**pip (ocrcpm)**](./pip/) | Depends on backend | Excellent | Medium | CPU/GPU | Document parsing as an installable SDK |

## Framework Details

### [vLLM](./vllm/) (Very Large Language Model)
- High-throughput inference engine with PagedAttention memory management
- Dynamic batching support, OpenAI-compatible API
- Ideal for production API services and large-scale batch inference
- Recommended hardware: GPU with more than 18GB of VRAM

### [SGLang](./sglang/) (Structured Generation Language)
- Structured generation optimization with efficient KV cache management
- Complex control flow and function calling optimization support
- Suitable for complex reasoning chains and structured text generation
- Recommended hardware: GPU with more than 18GB of VRAM

### [Ollama](./ollama/)
- One-click model management with simple command-line interface
- Automatic quantization support, REST API interface
- Perfect for personal development environments and research prototyping
- Hardware requirements: 8GB+ RAM, supports CPU/GPU

### [Llama.cpp](./llama.cpp/)
- Pure C++ implementation with CPU-optimized inference
- Multiple quantization support, lightweight deployment
- Ideal for mobile devices and edge computing
- Hardware requirements: 4GB+ RAM, various CPU architectures

### [pip (ocrcpm)](./pip/)
- Pip-installable OCR document-parsing pipeline: layout detection, crop recognition, Markdown assembly
- Command-line entry points plus a Python API; the inference engine stays pluggable
- Ideal for shipping OCR as a dependency rather than standing up a service by hand
- Hardware requirements: follow whichever backend it drives; layout detection runs on CPU

## Selection Guide

- **Production Environment (High Concurrency)**: vLLM - Best performance, optimal scalability
- **Complex Reasoning Tasks**: SGLang - Structured generation, function calling optimization
- **Personal Development**: Ollama - Simple to use, quick setup
- **Edge Deployment**: Llama.cpp - Lightweight, low power consumption
- **Document Parsing / OCR**: pip (ocrcpm) - One command from PDF or image to Markdown

## Hardware Requirements (cheat sheet)

Numbers below are **minimums for single-stream inference**. Add headroom
for higher batch sizes, longer contexts, multi-image inputs, or KV cache
on the activation side. Vision tower + KV cache typically add ~2 GB on top
of pure weight memory.

| Model              | Params | Precision         | Backend       | GPU VRAM     | CPU RAM    |
| :----------------- | :----- | :---------------- | :------------ | :---------:  | :--------: |
| **MiniCPM-V 4.6**  | 9B     | BF16 / FP16       | vLLM / SGLang | **≥ 20 GB**  |    —       |
|                    |        | AWQ / GPTQ (int4) | vLLM / SGLang | **≥ 9 GB**   |    —       |
|                    |        | BNB nf4 (int4)    | transformers  | **≥ 9 GB**   |    —       |
|                    |        | GGUF Q4_K_M       | llama.cpp / Ollama | ≥ 7 GB  | ≥ 8 GB     |
|                    |        | GGUF Q8_0         | llama.cpp / Ollama | ≥ 11 GB | ≥ 12 GB    |
| **MiniCPM-V 4.5**  | 8B     | BF16 / FP16       | vLLM / SGLang | **≥ 18 GB**  |    —       |
|                    |        | AWQ (int4)        | vLLM          | **≥ 8 GB**   |    —       |
|                    |        | GGUF Q4_K_M       | llama.cpp / Ollama | ≥ 6 GB  | ≥ 7 GB     |
| **MiniCPM-o 4.5**  | 9B     | BF16 / FP16       | vLLM          | **≥ 20 GB**  |    —       |
|                    |        | GGUF Q4_K_M       | llama.cpp     | ≥ 7 GB       | ≥ 8 GB     |
| **MiniCPM-V 4.0**  | 4B     | BF16 / FP16       | vLLM / SGLang | ≥ 10 GB      |    —       |
|                    |        | GGUF Q4_K_M       | llama.cpp / Ollama | ≥ 3 GB  | ≥ 4 GB     |

For exact numbers measured on specific hardware (A100, RTX 4090, M-series
Mac, …), see each model's HuggingFace card.

## Framework Support Matrix

For the up-to-date upstream merge status across all MiniCPM-V & o versions
(including v4.6 with `MiniCPMV4_6ForConditionalGeneration` in transformers,
the in-flight vLLM / SGLang PRs, etc.), see the
**[Framework Support Matrix in the root README](../README.md#framework-support-matrix)**.