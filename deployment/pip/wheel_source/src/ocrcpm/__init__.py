"""ocrcpm: MiniCPM-V OCR document-parsing pipeline.

A pipeline package (layout -> prepare -> infer) with a layered, OpenAI-compatible
inference design. Inspired by MinerU: the package itself only orchestrates the
pipeline and CLI. The SDK exposes local Transformers, http-client, and local
vLLM auto-start deployment modes; heavyweight inference engines remain opt-in
via pip extras.
"""

__version__ = "0.1.0"

from .config import load_config
from .pipeline import infer_run, parse_layout_run, prepare_run

__all__ = [
    "__version__",
    "load_config",
    "parse_layout_run",
    "prepare_run",
    "infer_run",
]
