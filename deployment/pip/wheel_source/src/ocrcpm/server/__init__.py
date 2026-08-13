"""Optional in-package inference backends.

The HF Transformers OpenAI-compatible server is provided as a fallback backend
for environments where the MiniCPM-V model is not yet natively supported by
vLLM. It exposes ``/v1/chat/completions`` so the SDK talks to it exactly like
any other OpenAI-compatible endpoint (http-client backend).
"""

__all__ = ["hf_openai_server"]
