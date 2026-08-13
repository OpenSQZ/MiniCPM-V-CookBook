#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

from PIL import Image

from ..config import load_config
from ..model_assets import DEFAULT_CONFIG_PATH
from ..pipeline import infer_run, parse_layout_run, prepare_run, read_markdown_from_run


DEFAULT_MODEL_NAME = "ocrcpm-pipeline"
DEFAULT_PAGE_DELIMITER = "\n\n<!-- page-break -->\n\n"


def _decode_data_url(url: str) -> Image.Image:
    if url.startswith("data:"):
        if "," not in url:
            raise ValueError("invalid image data URL")
        url = url.split(",", 1)[1]
    try:
        data = base64.b64decode(url, validate=True)
    except Exception as exc:
        raise ValueError("image_url must contain a base64 data URL") from exc
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ValueError("image_url does not contain a valid image") from exc


def _extract_request(payload: dict[str, Any]) -> tuple[Image.Image | None, str]:
    image = None
    texts: list[str] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            if message.get("role") == "user" and content:
                texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and message.get("role") == "user":
                text = str(item.get("text", "")).strip()
                if text:
                    texts.append(text)
            elif item.get("type") == "image_url":
                image_url = item.get("image_url") or {}
                url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                if url:
                    if image is not None:
                        image.close()
                    image = _decode_data_url(url)
    return image, "\n".join(texts).strip()


def _chat_response(
    *,
    text: str,
    model: str,
    request_id: str | None = None,
    created: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completion_tokens = len(text.split())
    response = {
        "id": request_id or "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
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
    }
    if metadata:
        response["pipeline_metrics"] = metadata
    return response


def _iter_sse_events(response: dict[str, Any], chunk_size: int = 2048) -> Iterator[bytes]:
    request_id = str(response["id"])
    created = int(response["created"])
    model = str(response["model"])
    text = str(response["choices"][0]["message"]["content"])

    def event(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
        body = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return ("data: " + json.dumps(body, ensure_ascii=False) + "\n\n").encode("utf-8")

    yield event({"role": "assistant"})
    for start in range(0, len(text), max(1, chunk_size)):
        yield event({"content": text[start : start + chunk_size]})
    yield event({}, "stop")
    yield b"data: [DONE]\n\n"


class OCRCPMPipelineServer:
    def __init__(
        self,
        config_path: str,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        run_root: str = "",
        page_delimiter: str = DEFAULT_PAGE_DELIMITER,
    ):
        self.config_path = str(Path(config_path).expanduser().resolve())
        if not Path(self.config_path).is_file():
            raise FileNotFoundError(f"pipeline config not found: {self.config_path}")
        self.model_name = model_name
        self.run_root = str(Path(run_root).expanduser().resolve()) if run_root else ""
        self.page_delimiter = page_delimiter
        self.lock = Lock()

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        image, _prompt = _extract_request(payload)
        requested_model = str(payload.get("model") or self.model_name)
        if image is None:
            return _chat_response(
                text="Please upload one document image for OCR.",
                model=requested_model,
            )

        request_id = uuid.uuid4().hex
        run_tag = f"openai_{time.strftime('%Y%m%d_%H%M%S')}_{request_id[:8]}"
        started = time.time()
        with tempfile.TemporaryDirectory(prefix="ocrcpm-openai-") as temp_dir:
            input_path = Path(temp_dir) / "input.png"
            try:
                image.save(input_path, format="PNG")
            finally:
                image.close()

            overrides: dict[str, Any] = {
                "run": {"run_tag": run_tag, "mode": "oneshot_file"},
                "input": {
                    "file_path": str(input_path),
                    "manifest": "",
                    "image_dir": "",
                    "files_dir": "",
                    "limit": 0,
                },
                "layout": {"enabled": True},
                "logging": {"progress_log_every": 10**9},
            }
            if self.run_root:
                overrides["run"]["run_root"] = self.run_root

            with self.lock:
                cfg = load_config(self.config_path, overrides=overrides)
                layout_summary = parse_layout_run(cfg)
                prepare_run(cfg)
                infer_summary = infer_run(cfg)
                run_dir = Path(cfg["_meta"]["run_dir"])
                markdown, pages = read_markdown_from_run(
                    run_dir,
                    merge_pages=True,
                    page_delimiter=self.page_delimiter,
                )

        return _chat_response(
            text=markdown,
            model=requested_model,
            metadata={
                "latency_sec": round(time.time() - started, 6),
                "pages": pages,
                "run_dir": str(run_dir),
                "layout_summary": layout_summary,
                "infer_summary": infer_summary,
            },
        )


def make_handler(server_state: OCRCPMPipelineServer):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_sse(self, response: dict[str, Any]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for event in _iter_sse_events(response):
                self.wfile.write(event)
                self.wfile.flush()

        def do_GET(self) -> None:
            if self.path in {"/health", "/"}:
                self._send_json(200, {"status": "ok", "service": "ocrcpm-pipeline"})
                return
            if self.path == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": server_state.model_name, "object": "model"}],
                    },
                )
                return
            self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                response = server_state.complete(payload)
                if payload.get("stream", False):
                    self._send_sse(response)
                else:
                    self._send_json(200, response)
            except Exception as exc:
                self._send_json(
                    500,
                    {"error": {"message": str(exc), "type": "pipeline_error"}},
                )

        def log_message(self, fmt: str, *args: Any) -> None:
            print("%s - %s" % (self.log_date_time_string(), fmt % args), flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible server for the full OCRCPM document pipeline"
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("OCRCPM_CONFIG", str(DEFAULT_CONFIG_PATH)),
        help="Pipeline YAML config; defaults to OCRCPM_CONFIG or the user config",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18600)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--run-root", default="")
    args = parser.parse_args()

    state = OCRCPMPipelineServer(
        args.config,
        model_name=args.model_name,
        run_root=args.run_root,
    )
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(
        f"OCRCPM pipeline OpenAI server listening on http://{args.host}:{args.port}",
        flush=True,
    )
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
