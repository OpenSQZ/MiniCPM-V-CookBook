# 推理示例

探索 MiniCPM-o 强大多模态能力的即用示例。

📖 [English Version](./README.md) | [返回主页](../)

## 环境准备

```bash
conda create -n minicpm python=3.10
conda activate minicpm
pip install -r requirements.txt 
```

## 可用示例

| 示例 | 描述 |
| ---- | ---- |
| [单图理解](./minicpm-v4_7_single_image.md) | 图像理解，包含物体检测和场景描述 |
| [多图对比](./minicpm-v4_7_multi_images.md) | 比较和分析多个图像的相似性与差异 |
| [OCR](./minicpm-v4_7_ocr.md) | 光学字符识别与版面分析 |
| [视觉定位](./minicpm-v4_7_grounding.md) | 视觉定位与目标框 |
| [PDF解析](./minicpm-v4_7_pdf_parse.md) | 解析PDF文档并结构化文本提取 |
| [视频理解](./minicpm-v4_7_video_understanding.md) | 视频内容分析和事件提取 |
| [场景文字识别](./minicpm-v4_5_scene_text_recognize.md) | 真实场景中的文字检测和识别 |
| [语音转文字](./speech2text.md) | 多语言语音识别 |
| [文字转语音](./text2speech.md) | 带有情感控制的语音合成 |
| [语音克隆](./voice_clone.md) | 提取声纹特征实现个性化语音合成 |
| [多模态RAG](./rag.md) | 检索增强的多模态知识问答 |
| [智能代理](./agent.md) | 具有工具使用能力的AI代理系统 | 