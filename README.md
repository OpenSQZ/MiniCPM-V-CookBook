

# 🍳 MiniCPM-V & o Cookbook

[🏠 Main Repository](https://github.com/OpenBMB/MiniCPM-o) | [📚 Full Documentation](https://opensqz.github.io/MiniCPM-V-CookBook/)

Cook up amazing AI applications effortlessly with MiniCPM-V, MiniCPM-o, and the MiniCPM LLM series — bringing text, vision, speech, and live-streaming capabilities right to your fingertips! For version-specific deployment instructions, see the files in the [deployment](./deployment/) directory.



## ✨ What Makes Our Recipes Special?

### Easy Usage Documentation

Our comprehensive [documentation website](https://opensqz.github.io/MiniCPM-V-CookBook/) presents every recipe in a clear, well-organized manner.
All features are displayed at a glance, making it easy for you to quickly find exactly what you need.

### Broad User Spectrum

We support a wide range of users, from individuals to enterprises and researchers.

- **Individuals**: Enjoy effortless inference using [Ollama](./deployment/ollama/minicpm-v4_6_ollama.md) and [Llama.cpp](./deployment/llama.cpp/minicpm-v4_6_llamacpp.md) with minimal setup.
- **Enterprises**: Achieve high-throughput, scalable performance with [vLLM](./deployment/vllm/minicpm-v4_6_vllm.md) and [SGLang](./deployment/sglang/minicpm-v4_6_sglang.md).
- **Researchers**: Leverage advanced frameworks including [Transformers](./finetune/official/finetune_full.md) , [LLaMA-Factory](./finetune/llama-factory/finetune_llamafactory.md), [SWIFT](./finetune/swift/swift.md), and [Align-anything](./finetune/align-anything/align_anything.md) to enable flexible model development and cutting-edge experimentation.

### Versatile Deployment Scenarios

Our ecosystem delivers optimal solution for a variety of hardware environments and deployment demands.

- **Web demo**: Launch interactive multimodal AI web demo with [FastAPI](./demo/README.md).
- **Quantized deployment**: Maximize efficiency and minimize resource consumption using [GGUF](./quantization/gguf/minicpm-v4_6_gguf_quantize.md), [BNB](./quantization/bnb/minicpm-v4_6_bnb_quantize.md), and [AWQ](./quantization/awq/minicpm-v4_6_awq_quantize.md).
- **Edge devices**: Local multimodal demos on **[MiniCPM-V-Apps](https://github.com/OpenBMB/MiniCPM-V-Apps)** (iOS / Android / HarmonyOS NEXT, `llama.cpp`); Cookbook iOS quickstart: [iPhone and iPad](./demo/ios_demo/ios.md).

## ⭐️ Live Demonstrations

Explore real-world examples of MiniCPM-V deployed on edge devices using our curated recipes. These demos highlight the model’s high efficiency and robust performance in practical scenarios.

- **[MiniCPM-V-Apps](https://github.com/OpenBMB/MiniCPM-V-Apps)** — on-device iOS / Android / HarmonyOS NEXT with `llama.cpp` (upstream [README](https://github.com/OpenBMB/MiniCPM-V-Apps/blob/main/README.md) · [README_zh](https://github.com/OpenBMB/MiniCPM-V-Apps/blob/main/README_zh.md) · [downloads](https://github.com/OpenBMB/MiniCPM-V-Apps/blob/main/DOWNLOAD_zh.md)). Cookbook focuses on the Xcode path: **[iOS demo](./demo/ios_demo/ios.md)**.





## 🔥 Inference Recipes

> *Ready-to-run examples*


| Recipe                                                          | Description                                        |
| --------------------------------------------------------------- | -------------------------------------------------- |
| **Vision Capabilities** (MiniCPM-V 4.6)                         |                                                    |
| 🖼️ [Single-image QA](./inference/minicpm-v4_6_single_image.md) | Question answering on a single image               |
| 🧩 [Multi-image QA](./inference/minicpm-v4_6_multi_images.md)   | Question answering with multiple images            |
| 🎬 [Video QA](./inference/minicpm-v4_6_video_understanding.md)  | Video-based question answering                     |
| 📄 [Document Parser](./inference/minicpm-v4_6_pdf_parse.md)     | Parse and extract content from PDFs and webpages   |
| 📝 [Text Recognition](./inference/minicpm-v4_6_ocr.md)          | Reliable OCR for photos and screenshots            |
| 🎯 [Grounding](./inference/minicpm-v4_6_grounding.md)           | Visual grounding and object localization in images |
| **Audio Capabilities** (MiniCPM-o)                              |                                                    |
| 🎤 [Speech-to-Text](./inference/speech2text.md)                 | Multilingual speech recognition                    |
| 🗣️ [Text-to-Speech](./inference/text2speech.md)                | Instruction-following speech synthesis             |
| 🎭 [Voice Cloning](./inference/voice_clone.md)                  | Realistic voice cloning and role-play              |
| **Text Capabilities** (MiniCPM LLM 5 / 4.1 / 4)                 |                                                    |
| 💬 [Chat & Hybrid Reasoning (MiniCPM 5)](./inference/minicpm5_chat.md) | Compact 1B LLM with Think / No-Think modes & tool calling |
| 💬 [Chat & Hybrid Reasoning (MiniCPM 4.1)](./inference/minicpm4_1_chat.md)    | 8B LLM chat with optional step-by-step reasoning      |


## 🏋️ Fine-tuning Recipes

> *Customize your model with your own ingredients*

**Data preparation**

Follow the [guidance](./finetune/official/dataset_guidance.md) to set up your training datasets.

**Training**

We provide training methods serving different needs as following:


| Framework                                            | Description                                        |
| ---------------------------------------------------- | -------------------------------------------------- |
| [Transformers](./finetune/official/finetune_full.md)          | Most flexible for customization                    |
| [LLaMA-Factory](./finetune/llama-factory/finetune_llamafactory.md) | Modular fine-tuning toolkit                        |
| [SWIFT](./finetune/swift/swift.md)                         | Lightweight and fast parameter-efficient tuning    |
| [Align-anything](./finetune/align-anything/align_anything.md)       | Visual instruction alignment for multimodal models |


## 📦 Serving Recipes

> *Deploy your model efficiently. Pick a framework — the cookbook docs page lets you switch between V / o / LLM versions on the sidebar.*


| Framework | Description                                                                                                                                  |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [vLLM](https://opensqz.github.io/MiniCPM-V-CookBook/en/v4.6/deployment/vllm.html)              | High-throughput GPU inference                                                                                |
| [SGLang](https://opensqz.github.io/MiniCPM-V-CookBook/en/v4.6/deployment/sglang.html)          | High-throughput GPU inference *(MiniCPM 5 on upstream `>=0.5.12` with native `minicpm5` tool-call parser; older LLM series via [`tc-mb/sglang`](https://github.com/tc-mb/sglang) fork)* |
| [llama.cpp](https://opensqz.github.io/MiniCPM-V-CookBook/en/v4.6/deployment/llamacpp.html)     | Fast CPU / GGUF inference on PC, iPhone and iPad                                                              |
| [Ollama](https://opensqz.github.io/MiniCPM-V-CookBook/en/v4.6/deployment/ollama.html)          | User-friendly one-line local run                                                                              |
| [MLX](https://opensqz.github.io/MiniCPM-V-CookBook/en/minicpm5/deployment/mlx.html)            | Apple Silicon inference                                                                                       |
| [OpenWebUI](./demo/web_demo/openwebui)                                                          | Interactive Web demo with Open WebUI                                                                          |
| [Gradio](./demo/web_demo/gradio/README.md)                                                      | Interactive Web demo with Gradio                                                                              |
| [FastAPI](./demo/README.md)                                                                     | Interactive Omni Streaming demo with FastAPI                                                                  |
| [iOS](./demo/ios_demo/ios.md)                                                                   | **[MiniCPM-V-Apps](https://github.com/OpenBMB/MiniCPM-V-Apps)** — iOS quickstart (`llama.cpp`; Android / HarmonyOS in upstream) |


## 🥄 Quantization Recipes

> *Compress your model to improve efficiency. The cookbook docs page covers all supported series — use the sidebar version switcher to pick a release.*


| Method                                                                                              | Key Feature                                                       |
| --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [GGUF](https://opensqz.github.io/MiniCPM-V-CookBook/en/v4.6/quantization/gguf.html)                 | Simplest and most portable format                                 |
| [BNB](https://opensqz.github.io/MiniCPM-V-CookBook/en/v4.6/quantization/bnb.html)                   | Simple and easy-to-use quantization method                        |
| [AWQ](https://opensqz.github.io/MiniCPM-V-CookBook/en/v4.6/quantization/awq.html)                   | High-performance INT4 quantization for efficient inference        |
| [GPTQ](https://opensqz.github.io/MiniCPM-V-CookBook/en/v4.5/quantization/gptq.html)                 | Weight-only INT4 with vLLM-compatible packaging (also supports QAT) |


## Awesome Works using MiniCPM-V & o

- [text-extract-api](https://github.com/CatchTheTornado/text-extract-api): Document extraction API using OCRs and Ollama supported models GitHub Repo stars
- [comfyui_LLM_party](https://github.com/heshengtao/comfyui_LLM_party): Build LLM workflows and integrate into existing image workflows GitHub Repo stars
- [Ollama-OCR](https://github.com/imanoop7/Ollama-OCR): OCR package uses vlms through Ollama to extract text from images and PDF GitHub Repo stars
- [comfyui-mixlab-nodes](https://github.com/MixLabPro/comfyui-mixlab-nodes): ComfyUI node suite supports Workflow-to-APP、GPT&3D and more GitHub Repo stars
- [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat): Interactive digital human conversation implementation on single PC GitHub Repo stars
- [pensieve](https://github.com/arkohut/pensieve): A privacy-focused passive recording project by recording screen content GitHub Repo stars
- [paperless-gpt](https://github.com/icereed/paperless-gpt): Use LLMs to handle paperless-ngx, AI-powered titles, tags and OCR GitHub Repo stars
- [Neuro](https://github.com/kimjammer/Neuro): A recreation of Neuro-Sama, but running on local models on consumer hardware GitHub Repo stars

## 👥 Community

### Contributing

We love new recipes! Please share your creative dishes:

1. Fork the repository
2. Create your recipe
3. Submit a pull request

### Issues & Support

- Found a bug? [Open an issue](https://github.com/OpenBMB/MiniCPM-o/issues)
- Need help? Join our [Discord](https://discord.gg/rftuRMbqzf)

## Institutions

This cookbook is developed by [OpenBMB](https://github.com/openbmb) and [OpenSQZ](https://github.com/opensqz).

## 📜 License

This cookbook is served under the [Apache-2.0 License](LICENSE) - cook freely, share generously! 🍳

## Citation 

If you find our model/code/paper helpful, please consider citing our papers 📝 and staring us ⭐️！

```bib
@misc{yu2025minicpmv45cookingefficient,
      title={MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipe}, 
      author={Tianyu Yu and Zefan Wang and Chongyi Wang and Fuwei Huang and Wenshuo Ma and Zhihui He and Tianchi Cai and Weize Chen and Yuxiang Huang and Yuanqian Zhao and Bokai Xu and Junbo Cui and Yingjing Xu and Liqing Ruan and Luoyuan Zhang and Hanyu Liu and Jingkun Tang and Hongyuan Liu and Qining Guo and Wenhao Hu and Bingxiang He and Jie Zhou and Jie Cai and Ji Qi and Zonghao Guo and Chi Chen and Guoyang Zeng and Yuxuan Li and Ganqu Cui and Ning Ding and Xu Han and Yuan Yao and Zhiyuan Liu and Maosong Sun},
      year={2025},
      eprint={2509.18154},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.18154}, 
}

@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={Nat Commun 16, 5509 (2025)},
  year={2025}
}

@article{cui2026minicpmo45realtimefullduplex,
      title={MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction},
      author={Junbo Cui and Bokai Xu and Chongyi Wang and Tianyu Yu and Weiyue Sun and Yingjing Xu and Tianran Wang and Zhihui He and Wenshuo Ma and Tianchi Cai and Jiancheng Gui and Luoyuan Zhang and Xian Sun and Fuwei Huang and Moye Chen and Zhuo Lin and Hanyu Liu and Qingxin Gui and Qingzhe Han and Yuyang Wen and Huiping Liu and Rongkang Wang and Yaqi Zhang and Hongliang Wei and Chi Chen and You Li and Kechen Fang and Jie Zhou and Yuxuan Li and Guoyang Zeng and Chaojun Xiao and Yankai Lin and Xu Han and Maosong Sun and Zhiyuan Liu and Yuan Yao},
      year={2026},
      eprint={2604.27393},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.27393},
}

```

