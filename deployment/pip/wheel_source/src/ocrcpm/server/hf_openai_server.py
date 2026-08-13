#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import time
import uuid
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from threading import Lock
from typing import Any

import torch
from PIL import Image


def _patch_transformers_attention() -> None:
    import transformers.modeling_utils

    original = transformers.modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation

    def patched(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except (ValueError, ImportError):
            return "eager"

    transformers.modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation = patched


def _decode_data_url(url: str) -> Image.Image:
    if "," in url:
        url = url.split(",", 1)[1]
    data = base64.b64decode(url)
    return Image.open(BytesIO(data)).convert("RGB")


def _extract_request(payload: dict[str, Any]) -> tuple[Image.Image | None, str]:
    messages = payload.get("messages") or []
    image = None
    texts: list[str] = []

    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                texts.append(str(item.get("text", "")))
            elif item.get("type") == "image_url":
                image_url = item.get("image_url") or {}
                url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                if url:
                    image = _decode_data_url(url)

    return image, "\n".join(t for t in texts if t).strip()


class MiniCPMVServer:
    def __init__(self, model_path: str):
        warnings.filterwarnings("ignore")
        _patch_transformers_attention()

        from transformers import AutoModel, AutoProcessor

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model_name = "minicpm-v4.7"
        self.lock = Lock()

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        image, prompt = _extract_request(payload)
        if image is None:
            raise ValueError("request must include one image_url")
        if not prompt:
            prompt = "Text Recognition:"

        mm_kwargs = payload.get("mm_processor_kwargs") or {}
        chat_kwargs = payload.get("chat_template_kwargs") or {}
        max_tokens = int(payload.get("max_tokens") or 256)
        temperature = float(payload.get("temperature", 0.01) or 0.01)
        top_p = float(payload.get("top_p", 0.001) or 0.001)
        top_k = int(payload.get("top_k", 1) or 1)
        repetition_penalty = float(payload.get("repetition_penalty", 1.0) or 1.0)
        no_repeat_ngram_size = int(payload.get("no_repeat_ngram_size", 0) or 0)

        with self.lock:
            started = time.time()
            text = self.model.chat(
                image=image,
                msgs=[{"role": "user", "content": prompt}],
                tokenizer=self.processor.tokenizer,
                processor=self.processor,
                max_new_tokens=max_tokens,
                sampling=True,
                temperature=max(temperature, 0.001),
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_slice_nums=int(mm_kwargs.get("max_slice_nums") or 9),
                enable_thinking=bool(chat_kwargs.get("enable_thinking", False)),
                downsample_mode=str(mm_kwargs.get("downsample_mode") or "4x"),
            )
            latency = time.time() - started

        completion_tokens = len(self.processor.tokenizer.encode(text))
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
            },
            "hf_metrics": {"latency_sec": latency},
        }


def make_handler(server_state: MiniCPMVServer):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path in {"/health", "/"}:
                self._send_json(200, {"status": "ok"})
                return
            if self.path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [{"id": server_state.model_name, "object": "model"}]})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._send_json(200, server_state.complete(payload))
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            print("%s - %s" % (self.log_date_time_string(), fmt % args), flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18599)
    args = parser.parse_args()

    state = MiniCPMVServer(args.model_path)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"HF MiniCPMV OpenAI server listening on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
