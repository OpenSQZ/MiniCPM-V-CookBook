# MiniCPM-V & o Cookbook

用 **MiniCPM-V**、**MiniCPM-o** 以及 **MiniCPM LLM** 系列轻松搭建强大的 AI 应用 —— 文本、视觉、语音、实时流媒体能力一站式获得。

## 最新动态

- 🎉 **MiniCPM-V 4.6** 发布 —— Instruct 与 Thinking 双版本，Qwen3.5 hybrid backbone，256K 上下文，重构后的视觉 merger 结构。
  - 推理：[单图问答](v4.6/inference/single-image.html) · [多图问答](v4.6/inference/multi-images.html) · [视频](v4.6/inference/video-understanding.html) · [OCR](v4.6/inference/ocr.html) · [PDF 解析](v4.6/inference/pdf-parse.html) · [Grounding](v4.6/inference/grounding.html)
  - 部署：[vLLM](v4.6/deployment/vllm.html) · [SGLang](v4.6/deployment/sglang.html) · [llama.cpp](v4.6/deployment/llamacpp.html) · [Ollama](v4.6/deployment/ollama.html)
  - 量化：[GGUF](v4.6/quantization/gguf.html) · [BNB](v4.6/quantization/bnb.html) · [AWQ](v4.6/quantization/awq.html)

## 按场景选择

### 个人用户

在自己机器上轻松推理 —— 支持 **CPU + GPU**、**macOS / Linux / Windows**，甚至手机。

- [Ollama](v4.6/deployment/ollama.html) —— 配置最简单
- [llama.cpp](v4.6/deployment/llamacpp.html) —— CPU 推理最快
- [iOS Demo](shared/demos/ios.html) —— 运行在 iPhone / iPad

### 企业用户

高吞吐、可规模化的服务化部署：

- [vLLM](v4.6/deployment/vllm.html) —— 生产级 GPU 推理
- [SGLang](v4.6/deployment/sglang.html) —— 高吞吐 GPU 推理

### 研究者

训练 / 微调 / 定制：

- [Transformers 全参 / LoRA](shared/finetune/full.html)
- [LLaMA-Factory](shared/finetune/llamafactory.html)
- [SWIFT](shared/finetune/swift.html)
- [Align-anything](shared/finetune/align-anything.html)

## 版本一览

本 Cookbook 覆盖目前在维护的所有 MiniCPM 版本：

### MiniCPM-V

| 版本 | 状态 | 模态 | 参数量 | 亮点 | 上下文 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4.6** *(最新)* | 推荐 | 图像、视频 | ~1.2B | 可上手机的多模态大模型，LLaVA-UHD v4 视觉塔 | 256K |
| 4.5 | 稳定 | 图像、视频 | 9B | 图像 + 视频理解，可选思考模式 | 32K |

### MiniCPM-o

| 版本 | 状态 | 模态 | 参数量 | 亮点 | 上下文 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4.5** *(最新)* | 推荐 | 图像、视频、音频 | 9B | 端到端全模态（视觉 + 语音 + TTS），全双工流式 | 32K |

### MiniCPM (LLM)

| 版本 | 状态 | 模态 | 参数量 | 亮点 | 上下文 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4.1** *(最新)* | 推荐 | 文本 | 8B | 混合思考、EAGLE3、InfLLM-V2 | 128K |
| 4 | 稳定 | 文本 | 0.5B / 8B | InfLLM-V2、FRSpec 投机解码 | 128K |
| SALA | 研究 | 文本 | 8B | 稀疏 + 线性混合注意力 | 1M+ |

侧边栏的 **版本切换器** 可以跳到对应版本的文档。

## 资源链接

- 🤗 [HuggingFace](https://huggingface.co/openbmb)
- 🤖 [魔搭社区](https://modelscope.cn/organization/OpenBMB)
- 📖 [技术博客](https://huggingface.co/papers/2509.18154)
- 💬 [Discord](https://discord.gg/rftuRMbqzf)
- 🐛 [反馈问题](https://github.com/OpenBMB/MiniCPM-o/issues)
