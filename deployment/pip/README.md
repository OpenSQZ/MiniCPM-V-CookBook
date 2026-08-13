# pip (ocrcpm)

`ocrcpm` is a pip-installable document-parsing SDK for MiniCPM-V OCR. It wraps the whole pipeline — layout detection, crop preparation, per-crop recognition and Markdown assembly — behind a single command line, and keeps the inference engine pluggable so the same pipeline runs on Transformers, vLLM, SGLang, llama.cpp or Ollama.

## Guides

| Model | English | 中文 |
| :--- | :---: | :---: |
| **MiniCPM-V OCR** | [Guide](./minicpm-v-ocr_pip.md) | [指南](./minicpm-v-ocr_pip_zh.md) |

## Contents

| Path | Description |
| :--- | :--- |
| `ocrcpm-0.1.0-py3-none-any.whl` | Prebuilt wheel, install with `pip install "./ocrcpm-0.1.0-py3-none-any.whl[all]"` |
| `wheel_source/` | Source behind that wheel — `pyproject.toml`, `MANIFEST.in` and `src/ocrcpm/`; rebuild with `python -m build` |

The OCR model weights are not part of the wheel and are not open-sourced yet; the guides use placeholder model paths for now.
