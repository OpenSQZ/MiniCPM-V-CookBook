from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .utils import ensure_dir


class EngineError(RuntimeError):
    pass


def _patch_transformers_attention() -> None:
    try:
        import transformers.modeling_utils
    except Exception:
        return

    original = transformers.modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation

    if getattr(original, "_ocrcpm_patched", False):
        return

    def patched(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except (ValueError, ImportError):
            return "eager"

    patched._ocrcpm_patched = True  # type: ignore[attr-defined]
    transformers.modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation = patched


def _torch_dtype_from_name(torch_module: Any, dtype_name: str):
    name = (dtype_name or "bfloat16").lower()
    mapping = {
        "auto": None,
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }
    if name not in mapping:
        raise EngineError(f"Unsupported engine.torch_dtype: {dtype_name}")
    return mapping[name]


def _repair_transformers_dynamic_module_cache(model_path: str, missing_file: str) -> bool:
    missing_path = Path(missing_file)
    if not missing_path.parent.exists():
        return False

    source_dir = Path(model_path)
    if not source_dir.exists():
        return False

    copied = False
    for py_file in source_dir.glob("*.py"):
        target = missing_path.parent / py_file.name
        if not target.exists():
            shutil.copy2(py_file, target)
            copied = True
    return copied


class TransformersLocalEngine:
    def __init__(self, cfg: Dict, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        self.engine_cfg = cfg["engine"]
        self.mm_cfg = cfg["mm"]
        self.decode_cfg = cfg["decode"]
        self.model = None
        self.processor = None
        self.torch = None
        self.lock = threading.Lock()

    def start(self) -> None:
        _patch_transformers_attention()
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except Exception as e:
            raise EngineError(f"Failed to import transformers local backend dependencies: {e}") from e

        self.torch = torch
        model_path = self.engine_cfg.get("model_path", "")
        if not model_path:
            raise EngineError("engine.model_path is required for minicpm_transformers_local")

        device_cfg = str(self.engine_cfg.get("device", "cuda")).lower()
        device = "cuda" if device_cfg.startswith("cuda") and torch.cuda.is_available() else "cpu"
        if device == "cuda":
            gpu_id = str(self.engine_cfg.get("gpu_id", "0"))
            try:
                torch.cuda.set_device(int(gpu_id))
            except ValueError:
                torch.cuda.set_device(gpu_id)

        dtype = _torch_dtype_from_name(torch, str(self.engine_cfg.get("torch_dtype", "bfloat16")))
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if dtype is not None:
            # Newer transformers prefers dtype; older versions accept torch_dtype.
            kwargs["torch_dtype"] = dtype

        try:
            self.model = AutoModel.from_pretrained(model_path, **kwargs)
        except FileNotFoundError as e:
            if not _repair_transformers_dynamic_module_cache(model_path, str(e.filename)):
                raise
            self.model = AutoModel.from_pretrained(model_path, **kwargs)
        if device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def close(self) -> None:
        self.model = None
        self.processor = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def predict_crop(self, image, prompt: str, max_new_tokens: int, slice_nums: Optional[int] = None) -> Dict:
        if self.model is None or self.processor is None:
            raise EngineError("Transformers local engine is not started")

        mm_slice_nums = int(slice_nums if slice_nums is not None else self.mm_cfg["max_slice_nums"])
        started = time.time()
        with self.lock:
            text = self.model.chat(
                image=image,
                msgs=[{"role": "user", "content": prompt}],
                tokenizer=self.processor.tokenizer,
                processor=self.processor,
                max_new_tokens=int(max_new_tokens),
                sampling=True,
                temperature=max(float(self.decode_cfg["temperature"]), 0.001),
                top_p=float(self.decode_cfg["top_p"]),
                top_k=int(self.decode_cfg["top_k"]),
                repetition_penalty=float(self.decode_cfg["repetition_penalty"]),
                no_repeat_ngram_size=int(self.decode_cfg["no_repeat_ngram_size"]),
                max_slice_nums=mm_slice_nums,
                enable_thinking=False,
                downsample_mode=str(self.mm_cfg.get("downsample_mode") or "4x"),
            )
        latency = time.time() - started
        output_tokens = len(self.processor.tokenizer.encode(text))
        return {
            "text": text,
            "finish_reason": "stop",
            "output_token_count": output_tokens,
            "request_metrics": {
                "latency_sec": latency,
                "prompt_tokens": 0,
                "completion_tokens": output_tokens,
                "total_tokens": output_tokens,
                "endpoint": "transformers-local",
            },
            "encoder_metrics": {},
            "raw_response": {"transformers_local": True},
        }


class VLLMOpenAIEngine:
    backend_label = "vLLM"

    def __init__(self, cfg: Dict, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        self.engine_cfg = cfg["engine"]
        self.mm_cfg = cfg["mm"]
        self.decode_cfg = cfg["decode"]
        self.model_name = self.engine_cfg["model_name"]
        server_urls = [str(u).strip() for u in self.engine_cfg.get("server_urls", []) if str(u).strip()]
        server_url_single = str(self.engine_cfg.get("server_url", "")).strip()
        if server_url_single:
            server_urls.insert(0, server_url_single)
        if not server_urls:
            server_urls = ["http://127.0.0.1:%s" % self.engine_cfg.get("port")]
        self.server_urls = [u.rstrip("/") for u in server_urls]
        self.server_url = self.server_urls[0]
        self.external_server_pool = bool(self.engine_cfg.get("server_urls"))
        self.proc: Optional[subprocess.Popen] = None

        self.http = requests.Session()
        self.http.trust_env = False
        self._restart_lock = threading.Lock()

    def start(self) -> None:
        if self.external_server_pool:
            self._wait_ready(timeout_sec=int(self.engine_cfg.get("startup_timeout_sec", 1800)))
            return

        if not self.engine_cfg.get("auto_start", True):
            self._wait_ready(timeout_sec=int(self.engine_cfg.get("startup_timeout_sec", 1800)))
            return

        log_path = self.run_dir / "logs" / "engine.log"
        ensure_dir(log_path.parent)

        python_bin = self.engine_cfg.get("vllm_python", "")
        if not python_bin:
            raise EngineError(
                "engine.vllm_python is not set. To auto-start a local vLLM server, "
                "set engine.vllm_python in the config or the OCRCPM_VLLM_PYTHON "
                "environment variable. Alternatively, point engine.server_urls at an "
                "already-running OpenAI-compatible endpoint (http-client backend)."
            )
        cmd = [
            python_bin,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.engine_cfg["model_path"],
            "--served-model-name",
            self.model_name,
            "--trust-remote-code",
            "--port",
            str(self.engine_cfg["port"]),
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            str(self.engine_cfg["max_model_len"]),
            "--max-num-batched-tokens",
            str(self.engine_cfg["max_num_batched_tokens"]),
            "--limit-mm-per-prompt",
            json.dumps({"image": int(self.mm_cfg["image_limit"])}),
            "--gpu-memory-utilization",
            str(self.engine_cfg["gpu_memory_utilization"]),
            "--enforce-eager",
        ]

        env = os.environ.copy()
        cuda_home = self.engine_cfg.get("cuda_home", "")
        if cuda_home:
            env["CUDA_HOME"] = cuda_home
            env["PATH"] = "{}/bin:{}:{}".format(cuda_home, Path(python_bin).parent, env.get("PATH", ""))
        # Preload conda's libstdc++ for pre-compiled vLLM .so compatibility
        conda_lib = Path(python_bin).parent.parent / "lib" / "libstdc++.so.6"
        if conda_lib.exists():
            existing = env.get("LD_PRELOAD", "")
            env["LD_PRELOAD"] = str(conda_lib) if not existing else f"{conda_lib}:{existing}"
        env["CUDA_VISIBLE_DEVICES"] = str(self.engine_cfg.get("gpu_id", "0"))
        env["no_proxy"] = "{},127.0.0.1,localhost,::1".format(env.get("no_proxy", "")).strip(",")
        env["NO_PROXY"] = "{},127.0.0.1,localhost,::1".format(env.get("NO_PROXY", "")).strip(",")

        # Keep append mode so restart cycles do not truncate prior logs.
        with log_path.open("a", encoding="utf-8") as f:
            self.proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

        if bool(self.engine_cfg.get("wait_ready", False)):
            self._wait_ready(timeout_sec=int(self.engine_cfg.get("startup_timeout_sec", 1800)))
        else:
            time.sleep(float(self.engine_cfg.get("startup_sleep_sec", 3)))
            if self.proc and self.proc.poll() is not None:
                raise EngineError("vLLM exited early with code=%s" % self.proc.returncode)

    def _wait_ready(self, timeout_sec: int = 1800) -> None:
        deadline = time.time() + timeout_sec
        last_err = ""
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise EngineError("%s process exited early with code=%s" % (self.backend_label, self.proc.returncode))
            for base_url in self.server_urls:
                try:
                    resp = self.http.get("{}/v1/models".format(base_url), timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    ids = [item.get("id") for item in data.get("data", []) if item.get("id")]
                    if ids:
                        return
                    last_err = "{}/v1/models returned empty model list".format(base_url)
                except Exception as e:
                    last_err = "{}: {}".format(base_url, e)
            time.sleep(5)
        raise EngineError("%s startup timeout (%ss): %s" % (self.backend_label, timeout_sec, last_err))

    def _restart_local_server(self, reason: str) -> None:
        if self.external_server_pool:
            raise EngineError("Engine restart is disabled in external server pool mode")
        if not self.engine_cfg.get("auto_start", True):
            raise EngineError("Engine restart is disabled because auto_start=false")

        restart_log = self.run_dir / "logs" / "engine_restart.log"
        ensure_dir(restart_log.parent)
        with restart_log.open("a", encoding="utf-8") as f:
            f.write(
                "%s restart_reason=%s\n"
                % (__import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z", reason)
            )

        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)

        self.proc = None
        self.start()

    def _restart_local_server_if_needed(self, reason: str) -> None:
        with self._restart_lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            self._restart_local_server(reason)

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None
        self.http.close()

    def _sampling_params(self) -> Dict[str, Any]:
        return {
            "temperature": self.decode_cfg["temperature"],
            "top_p": self.decode_cfg["top_p"],
            "top_k": self.decode_cfg["top_k"],
            "repetition_penalty": self.decode_cfg["repetition_penalty"],
            "no_repeat_ngram_size": self.decode_cfg["no_repeat_ngram_size"],
        }

    def _message_content(self, img_b64: str, prompt: str) -> list[Dict[str, Any]]:
        return [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,%s" % img_b64}},
            {"type": "text", "text": prompt},
        ]

    def _mm_processor_kwargs(self, slice_nums: Optional[int]) -> Dict[str, Any]:
        return {
            "downsample_mode": self.mm_cfg["downsample_mode"],
            "max_slice_nums": int(
                slice_nums
                if slice_nums is not None
                else self.mm_cfg["max_slice_nums"]
            ),
        }

    def _chat_url(self, base_url: str) -> str:
        return "{}/v1/chat/completions".format(base_url)

    def _build_payload(
        self,
        img_b64: str,
        prompt: str,
        max_tokens: int,
        slice_nums: Optional[int],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": self._message_content(img_b64, prompt),
                }
            ],
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": {"enable_thinking": False},
            **self._sampling_params(),
        }
        mm_processor_kwargs = self._mm_processor_kwargs(slice_nums)
        if mm_processor_kwargs:
            payload["mm_processor_kwargs"] = mm_processor_kwargs
        return payload

    def _payload_max_tokens(self, payload: Dict[str, Any]) -> int:
        return int(payload.get("max_tokens", 0) or 0)

    def _set_payload_max_tokens(self, payload: Dict[str, Any], value: int) -> None:
        payload["max_tokens"] = int(value)

    def _parse_response(self, data: Dict[str, Any], latency: float, endpoint: str) -> Dict:
        if "error" in data:
            raise EngineError(str(data["error"]))
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return {
            "text": choice["message"]["content"],
            "finish_reason": choice.get("finish_reason"),
            "output_token_count": int(usage.get("completion_tokens", 0)),
            "request_metrics": {
                "latency_sec": latency,
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "endpoint": endpoint,
            },
            "encoder_metrics": {},
            "raw_response": data,
        }

    def predict_crop(self, image, prompt: str, max_new_tokens: int, slice_nums: Optional[int] = None) -> Dict:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        requested_max_tokens = int(max_new_tokens)
        model_max_len = int(self.engine_cfg.get("max_model_len", 0) or 0)
        prompt_reserve_tokens = int(self.engine_cfg.get("prompt_reserve_tokens", 512) or 512)
        if model_max_len > 0:
            safe_ceiling = max(256, model_max_len - max(1, prompt_reserve_tokens))
            requested_max_tokens = min(requested_max_tokens, safe_ceiling)

        payload = self._build_payload(img_b64, prompt, requested_max_tokens, slice_nums)

        ready_timeout_sec = float(self.engine_cfg.get("request_ready_timeout_sec", 600))
        retry_interval_sec = float(self.engine_cfg.get("request_retry_interval_sec", 2))
        deadline = time.time() + ready_timeout_sec
        last_err = ""

        auto_restart = bool(self.engine_cfg.get("auto_restart_on_engine_dead", True))
        restart_budget = int(self.engine_cfg.get("max_restart_attempts_per_request", 1) or 1)

        request_http = requests.Session()
        request_http.trust_env = False
        try:
            while True:
                if self.proc and self.proc.poll() is not None:
                    if auto_restart and (not self.external_server_pool) and restart_budget > 0:
                        exit_code = self.proc.returncode
                        restart_budget -= 1
                        self._restart_local_server_if_needed("process_exited_before_request code=%s" % exit_code)
                        continue
                    raise EngineError(
                        "%s process exited during inference with code=%s"
                        % (self.backend_label, self.proc.returncode)
                    )

                try:
                    request_urls = list(self.server_urls)
                    random.shuffle(request_urls)

                    last_endpoint_err = ""
                    should_retry_with_lower_tokens = False
                    for target_url in request_urls:
                        try:
                            started = time.time()
                            resp = request_http.post(
                                self._chat_url(target_url),
                                json=payload,
                                timeout=float(self.engine_cfg.get("request_timeout_sec", 300)),
                            )
                            latency = time.time() - started

                            if resp.status_code >= 400:
                                body = resp.text or ""
                                current_max_tokens = self._payload_max_tokens(payload)
                                if (
                                    resp.status_code == 400
                                    and "maximum context length" in body
                                    and current_max_tokens > 256
                                ):
                                    lowered = max(256, current_max_tokens // 2)
                                    self._set_payload_max_tokens(payload, lowered)
                                    last_endpoint_err = "{} context overflow, retry with max_tokens={}: {}".format(
                                        target_url,
                                        lowered,
                                        body[:300],
                                    )
                                    should_retry_with_lower_tokens = True
                                    break
                                resp.raise_for_status()

                            return self._parse_response(resp.json(), latency, target_url)
                        except Exception as endpoint_err:
                            last_endpoint_err = "{}: {}".format(target_url, endpoint_err)
                            continue

                    if should_retry_with_lower_tokens:
                        last_err = last_endpoint_err
                        continue

                    raise EngineError(last_endpoint_err or "all endpoints failed")
                except Exception as e:
                    last_err = str(e)
                    lower_err = last_err.lower()
                    should_restart = (
                        auto_restart
                        and (not self.external_server_pool)
                        and restart_budget > 0
                        and self.proc is not None
                        and (
                            self.proc.poll() is not None
                            or "enginedeaderror" in lower_err
                            or "enginecore encountered an issue" in lower_err
                        )
                    )
                    if should_restart:
                        restart_budget -= 1
                        try:
                            code = self.proc.returncode if self.proc else "unknown"
                            self._restart_local_server_if_needed("request_exception=%s code=%s" % (last_err[:200], code))
                            continue
                        except Exception as restart_err:
                            last_err = "%s; restart_failed=%s" % (last_err, restart_err)
                    if time.time() >= deadline:
                        raise EngineError(
                            "%s request failed until timeout (%ss): %s"
                            % (self.backend_label, ready_timeout_sec, last_err)
                        )
                    time.sleep(retry_interval_sec)
        finally:
            request_http.close()


class LlamaCppOpenAIEngine(VLLMOpenAIEngine):
    backend_label = "llama.cpp"

    def start(self) -> None:
        if self.external_server_pool:
            self._wait_ready(timeout_sec=int(self.engine_cfg.get("startup_timeout_sec", 1800)))
            return

        if not self.engine_cfg.get("auto_start", True):
            self._wait_ready(timeout_sec=int(self.engine_cfg.get("startup_timeout_sec", 1800)))
            return

        llama_server = str(
            self.engine_cfg.get("llama_server")
            or os.environ.get("OCRCPM_LLAMA_SERVER", "")
        ).strip()
        if not llama_server:
            raise EngineError(
                "engine.llama_server is required to auto-start llama.cpp. "
                "Set it in the config or OCRCPM_LLAMA_SERVER, or set "
                "auto_start=false and point server_urls at an existing service."
            )

        model_path = str(self.engine_cfg.get("model_path", "")).strip()
        mmproj_path = str(self.engine_cfg.get("mmproj_path", "")).strip()
        if not model_path:
            raise EngineError("engine.model_path must point to the language-model GGUF")
        if not mmproj_path:
            raise EngineError("engine.mmproj_path must point to the multimodal-projector GGUF")

        log_path = self.run_dir / "logs" / "engine.log"
        ensure_dir(log_path.parent)

        cmd = [
            llama_server,
            "-m",
            model_path,
            "--mmproj",
            mmproj_path,
            "-ngl",
            str(self.engine_cfg.get("gpu_layers", 99)),
            "-c",
            str(self.engine_cfg.get("max_model_len", 8192)),
            "-np",
            str(self.engine_cfg.get("parallel", 1)),
            "--repeat-penalty",
            str(self.decode_cfg.get("repetition_penalty", 1.1)),
            "--host",
            str(self.engine_cfg.get("host", "127.0.0.1")),
            "--port",
            str(self.engine_cfg.get("port", 18499)),
            "--jinja",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.engine_cfg.get("gpu_id", "0"))
        env["no_proxy"] = "{},127.0.0.1,localhost,::1".format(env.get("no_proxy", "")).strip(",")
        env["NO_PROXY"] = "{},127.0.0.1,localhost,::1".format(env.get("NO_PROXY", "")).strip(",")

        with log_path.open("a", encoding="utf-8") as f:
            self.proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

        if bool(self.engine_cfg.get("wait_ready", True)):
            self._wait_ready(timeout_sec=int(self.engine_cfg.get("startup_timeout_sec", 1800)))
        else:
            time.sleep(float(self.engine_cfg.get("startup_sleep_sec", 3)))
            if self.proc and self.proc.poll() is not None:
                raise EngineError("llama.cpp exited early with code=%s" % self.proc.returncode)

    def _sampling_params(self) -> Dict[str, Any]:
        # llama-server uses its native repeat_penalty key. The vLLM-compatible
        # repetition_penalty key is accepted as unknown JSON but is not applied.
        return {
            "temperature": self.decode_cfg["temperature"],
            "top_p": self.decode_cfg["top_p"],
            "top_k": self.decode_cfg["top_k"],
            "repeat_penalty": self.decode_cfg["repetition_penalty"],
        }


class OllamaEngine(VLLMOpenAIEngine):
    """Ollama backend built on Ollama's native chat API.

    Ollama's OpenAI-compatible endpoint silently drops top_k and repeat_penalty,
    which OCR decoding depends on, so requests go to /api/chat where the whole
    sampler is configurable through the options object.
    """

    backend_label = "Ollama"

    def _ollama_bin(self) -> str:
        return (
            str(self.engine_cfg.get("ollama_bin") or "").strip()
            or os.environ.get("OCRCPM_OLLAMA_BIN", "").strip()
            or "ollama"
        )

    def _num_ctx(self) -> int:
        return int(self.engine_cfg.get("max_model_len", 8192) or 8192)

    def _model_tag(self) -> str:
        name = str(self.model_name).strip()
        return name if ":" in name else "{}:latest".format(name)

    def _chat_url(self, base_url: str) -> str:
        return "{}/api/chat".format(base_url)

    def _sampling_params(self) -> Dict[str, Any]:
        # Ollama names the repetition control repeat_penalty and has no
        # equivalent of no_repeat_ngram_size.
        return {
            "temperature": self.decode_cfg["temperature"],
            "top_p": self.decode_cfg["top_p"],
            "top_k": self.decode_cfg["top_k"],
            "repeat_penalty": self.decode_cfg["repetition_penalty"],
        }

    def _build_payload(
        self,
        img_b64: str,
        prompt: str,
        max_tokens: int,
        slice_nums: Optional[int],
    ) -> Dict[str, Any]:
        # MiniCPM-V4.7's downsample mode and slice count are baked into the GGUF
        # pair at conversion time, so there is no per-request mm knob to send.
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
            "stream": False,
            "think": False,
            # Passed through as configured: Ollama accepts a duration string as
            # well as a number of seconds, where a negative value means forever.
            "keep_alive": self.engine_cfg.get("keep_alive", "30m"),
            "options": {
                "num_predict": int(max_tokens),
                "num_ctx": self._num_ctx(),
                "num_gpu": int(self.engine_cfg.get("gpu_layers", 99)),
                **self._sampling_params(),
            },
        }

    def _payload_max_tokens(self, payload: Dict[str, Any]) -> int:
        return int(payload.get("options", {}).get("num_predict", 0) or 0)

    def _set_payload_max_tokens(self, payload: Dict[str, Any], value: int) -> None:
        payload.setdefault("options", {})["num_predict"] = int(value)

    def _parse_response(self, data: Dict[str, Any], latency: float, endpoint: str) -> Dict:
        if data.get("error"):
            raise EngineError(str(data["error"]))
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        return {
            "text": (data.get("message") or {}).get("content", ""),
            "finish_reason": data.get("done_reason"),
            "output_token_count": completion_tokens,
            "request_metrics": {
                "latency_sec": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "endpoint": endpoint,
            },
            "encoder_metrics": {},
            "raw_response": data,
        }

    def _client_env(self, base_url: str) -> Dict[str, str]:
        env = os.environ.copy()
        env["OLLAMA_HOST"] = base_url
        env["no_proxy"] = "{},127.0.0.1,localhost,::1".format(env.get("no_proxy", "")).strip(",")
        env["NO_PROXY"] = "{},127.0.0.1,localhost,::1".format(env.get("NO_PROXY", "")).strip(",")
        return env

    def start(self) -> None:
        if self.external_server_pool or not self.engine_cfg.get("auto_start", True):
            self._wait_ready(timeout_sec=int(self.engine_cfg.get("startup_timeout_sec", 1800)))
            return

        log_path = self.run_dir / "logs" / "engine.log"
        ensure_dir(log_path.parent)

        env = self._client_env(
            "{}:{}".format(
                self.engine_cfg.get("host", "127.0.0.1"),
                self.engine_cfg.get("port", 11434),
            )
        )
        env["CUDA_VISIBLE_DEVICES"] = str(self.engine_cfg.get("gpu_id", "0"))
        # Match the per-request num_ctx so the first request does not force the
        # scheduler to drop and reload the runner with different options.
        env["OLLAMA_CONTEXT_LENGTH"] = str(self._num_ctx())
        env["OLLAMA_NUM_PARALLEL"] = str(self.engine_cfg.get("parallel", 1))
        env["OLLAMA_KEEP_ALIVE"] = str(self.engine_cfg.get("keep_alive", "30m"))
        models_dir = str(self.engine_cfg.get("ollama_models_dir", "")).strip()
        if models_dir:
            env["OLLAMA_MODELS"] = models_dir

        with log_path.open("a", encoding="utf-8") as f:
            self.proc = subprocess.Popen(
                [self._ollama_bin(), "serve"],
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )

        if bool(self.engine_cfg.get("wait_ready", True)):
            self._wait_ready(timeout_sec=int(self.engine_cfg.get("startup_timeout_sec", 1800)))
        else:
            time.sleep(float(self.engine_cfg.get("startup_sleep_sec", 3)))
            if self.proc and self.proc.poll() is not None:
                raise EngineError("Ollama exited early with code=%s" % self.proc.returncode)

    def _wait_ready(self, timeout_sec: int = 1800) -> None:
        deadline = time.time() + timeout_sec
        last_err = ""
        reachable_url = ""
        while not reachable_url and time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise EngineError(
                    "%s process exited early with code=%s" % (self.backend_label, self.proc.returncode)
                )
            for base_url in self.server_urls:
                try:
                    self._installed_tags(base_url)
                    reachable_url = base_url
                    break
                except Exception as e:
                    last_err = "{}: {}".format(base_url, e)
            if not reachable_url:
                time.sleep(5)

        if not reachable_url:
            raise EngineError("%s startup timeout (%ss): %s" % (self.backend_label, timeout_sec, last_err))
        self._ensure_model_installed(reachable_url)

    def _installed_tags(self, base_url: str) -> set:
        resp = self.http.get("{}/api/tags".format(base_url), timeout=30)
        resp.raise_for_status()
        tags = set()
        for item in resp.json().get("models") or []:
            for key in ("name", "model"):
                value = item.get(key)
                if value:
                    tags.add(value if ":" in value else "{}:latest".format(value))
        return tags

    def _ensure_model_installed(self, base_url: str) -> None:
        tag = self._model_tag()
        if tag in self._installed_tags(base_url):
            return

        model_path = str(self.engine_cfg.get("model_path", "")).strip()
        mmproj_path = str(self.engine_cfg.get("mmproj_path", "")).strip()
        if not model_path or not mmproj_path:
            raise EngineError(
                "Ollama model %r is not registered, and engine.model_path / engine.mmproj_path "
                "are not both set so it cannot be created automatically. Either register it once "
                "with `ollama create %s -f Modelfile`, or point both settings at the MiniCPM-V4.7 "
                "GGUF pair." % (tag, self.model_name)
            )

        self._create_model(base_url, model_path, mmproj_path)
        if tag not in self._installed_tags(base_url):
            raise EngineError("`ollama create %s` reported success but %r is still missing" % (self.model_name, tag))

    def _create_model(self, base_url: str, model_path: str, mmproj_path: str) -> None:
        for label, path in (("engine.model_path", model_path), ("engine.mmproj_path", mmproj_path)):
            if not Path(path).exists():
                raise EngineError("{} not found: {}".format(label, path))

        modelfile = self.run_dir / "ollama" / "Modelfile"
        ensure_dir(modelfile.parent)
        modelfile.write_text("FROM {}\nFROM {}\n".format(model_path, mmproj_path), encoding="utf-8")

        log_path = self.run_dir / "logs" / "engine.log"
        ensure_dir(log_path.parent)
        timeout_sec = float(self.engine_cfg.get("model_create_timeout_sec", 3600))
        with log_path.open("a", encoding="utf-8") as f:
            f.write("ollama create {} -f {}\n".format(self.model_name, modelfile))
            f.flush()
            try:
                completed = subprocess.run(
                    [self._ollama_bin(), "create", self.model_name, "-f", str(modelfile)],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=self._client_env(base_url),
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired as e:
                raise EngineError(
                    "`ollama create %s` timed out after %ss; see %s"
                    % (self.model_name, timeout_sec, log_path)
                ) from e
        if completed.returncode != 0:
            raise EngineError(
                "`ollama create %s` failed with code=%s; see %s"
                % (self.model_name, completed.returncode, log_path)
            )

    def close(self) -> None:
        # A server we started is torn down with the process. For an external
        # server the model would otherwise sit in VRAM until keep_alive expires.
        if self.proc is None and bool(self.engine_cfg.get("unload_after_run", True)):
            try:
                self.http.post(
                    self._chat_url(self.server_url),
                    json={"model": self.model_name, "messages": [], "keep_alive": 0},
                    timeout=30,
                )
            except Exception:
                pass
        super().close()


class SGLangOpenAIEngine(VLLMOpenAIEngine):
    backend_label = "SGLang"

    def _message_content(self, img_b64: str, prompt: str) -> list[Dict[str, Any]]:
        # SGLang's MiniCPM-V processor determines visual placeholder sizes while
        # parsing the text prefix, so the text part must precede the image part.
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,%s" % img_b64}},
        ]

    def _mm_processor_kwargs(self, slice_nums: Optional[int]) -> Dict[str, Any]:
        # The SGLang V4.7 processor gets 4x mode from the server override and
        # uses the checkpoint's slice settings. Passing vLLM's request-specific
        # kwargs causes it to produce zero visual placeholder tokens.
        return {}

    def start(self) -> None:
        if self.external_server_pool:
            self._wait_ready(timeout_sec=int(self.engine_cfg.get("startup_timeout_sec", 1800)))
            return

        if not self.engine_cfg.get("auto_start", True):
            self._wait_ready(timeout_sec=int(self.engine_cfg.get("startup_timeout_sec", 1800)))
            return

        python_bin = str(
            self.engine_cfg.get("sglang_python")
            or os.environ.get("OCRCPM_SGLANG_PYTHON", "")
        ).strip()
        if not python_bin:
            raise EngineError(
                "engine.sglang_python is required to auto-start SGLang. "
                "Set it in the config or OCRCPM_SGLANG_PYTHON, or set "
                "auto_start=false and point server_urls at an existing service."
            )

        model_path = str(self.engine_cfg.get("model_path", "")).strip()
        if not model_path:
            raise EngineError("engine.model_path is required for the SGLang backend")

        log_path = self.run_dir / "logs" / "engine.log"
        ensure_dir(log_path.parent)

        cmd = [
            python_bin,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--served-model-name",
            self.model_name,
            "--trust-remote-code",
            "--dtype",
            str(self.engine_cfg.get("torch_dtype", "bfloat16")),
            "--context-length",
            str(self.engine_cfg.get("max_model_len", 8192)),
            "--mem-fraction-static",
            str(self.engine_cfg.get("gpu_memory_utilization", 0.8)),
            "--json-model-override-args",
            json.dumps({"downsample_mode": self.mm_cfg.get("downsample_mode", "4x")}),
            "--host",
            str(self.engine_cfg.get("host", "127.0.0.1")),
            "--port",
            str(self.engine_cfg.get("port", 18499)),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.engine_cfg.get("gpu_id", "0"))
        env["PYTHONNOUSERSITE"] = "1"
        env["no_proxy"] = "{},127.0.0.1,localhost,::1".format(env.get("no_proxy", "")).strip(",")
        env["NO_PROXY"] = "{},127.0.0.1,localhost,::1".format(env.get("NO_PROXY", "")).strip(",")

        cuda_home = str(self.engine_cfg.get("cuda_home", "")).strip()
        if cuda_home:
            env["CUDA_HOME"] = cuda_home
            env["PATH"] = "{}:{}:{}".format(
                Path(cuda_home) / "bin",
                Path(python_bin).parent,
                env.get("PATH", ""),
            )
            cuda_runtime_lib = Path(cuda_home) / "targets" / "x86_64-linux" / "lib"
            if cuda_runtime_lib.exists():
                for name in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
                    existing = env.get(name, "")
                    env[name] = (
                        str(cuda_runtime_lib)
                        if not existing
                        else f"{cuda_runtime_lib}:{existing}"
                    )

        jit_cache_dir = str(self.engine_cfg.get("sglang_jit_cache_dir", "")).strip()
        if not jit_cache_dir:
            cache_key = hashlib.sha256(cuda_home.encode("utf-8")).hexdigest()[:12]
            jit_cache_dir = str(
                Path.home() / ".cache" / "ocrcpm" / "sglang-tvm-ffi" / cache_key
            )
        env["TVM_FFI_CACHE_DIR"] = jit_cache_dir

        with log_path.open("a", encoding="utf-8") as f:
            self.proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )

        if bool(self.engine_cfg.get("wait_ready", True)):
            self._wait_ready(timeout_sec=int(self.engine_cfg.get("startup_timeout_sec", 1800)))
        else:
            time.sleep(float(self.engine_cfg.get("startup_sleep_sec", 3)))
            if self.proc and self.proc.poll() is not None:
                raise EngineError("SGLang exited early with code=%s" % self.proc.returncode)


def create_engine(cfg: Dict, run_dir: Path):
    engine_type = cfg["engine"].get("type", "minicpm_vllm_openai_api")
    if engine_type == "minicpm_transformers_local":
        return TransformersLocalEngine(cfg, run_dir)
    if engine_type == "minicpm_vllm_openai_api":
        return VLLMOpenAIEngine(cfg, run_dir)
    if engine_type == "minicpm_sglang_openai_api":
        return SGLangOpenAIEngine(cfg, run_dir)
    if engine_type == "minicpm_llamacpp_openai_api":
        return LlamaCppOpenAIEngine(cfg, run_dir)
    if engine_type == "minicpm_ollama_api":
        return OllamaEngine(cfg, run_dir)
    raise EngineError(f"Unsupported engine type: {engine_type}")
