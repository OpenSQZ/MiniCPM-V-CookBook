from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "run": {
        "run_tag": "",
        "run_root": "./runs",
        "mode": "smoke_1page",
        "seed": 42,
    },
    "input": {
        "manifest": "",
        "file_path": "",
        "image_dir": "",
        "files_dir": "",
        "limit": 0,
    },
    "engine": {
        "type": "minicpm_vllm_openai_api",
        "model_path": "",
        "server_url": "",
        "server_urls": [],
        "model_name": "minicpm-v4.7",
        "device": "cuda",
        "torch_dtype": "bfloat16",
        "auto_start": True,
        "vllm_python": "",
        "sglang_python": "",
        "sglang_jit_cache_dir": "",
        "llama_server": "",
        "mmproj_path": "",
        "ollama_bin": "",
        "ollama_models_dir": "",
        "keep_alive": "30m",
        "host": "127.0.0.1",
        "gpu_layers": 99,
        "parallel": 1,
        "cuda_home": "",
        "gpu_id": "0",
        "port": 18499,
        "max_model_len": 40960,
        "max_num_batched_tokens": 40960,
        "image_limit": 36,
        "gpu_memory_utilization": 0.9,
        "startup_timeout_sec": 1800,
        "auto_restart_on_engine_dead": True,
        "max_restart_attempts_per_request": 1,
    },
    "layout": {
        "enabled": True,
        "provider": "ppdoclayout3",
        "model_dir": "",
        "device": "cuda",
        "batch_size": 1,
    },
    "pipeline": {
        "max_workers": 32,
        "page_workers": 1,
        "page_maxsize": 0,
        "region_maxsize": 0,
        "queue_mode": "serial",
    },
    "mm": {
        "downsample_mode": "4x",
        "max_slice_nums": 9,
        "image_limit": 36,
    },
    "prompt": {
        "override_mode": "legacy",
        "prompt_map": "",
    },
    "decode": {
        "temperature": 0.01,
        "top_p": 0.001,
        "top_k": 1,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 100,
        "max_tokens_map": {},
    },
    "logging": {
        "level": "INFO",
        "progress_log_every": 20,
    },
}


ALLOWED_FILE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_path(base_dir: Path, value: str) -> str:
    if not value:
        return ""
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _validate_config(cfg: Dict[str, Any]) -> None:
    engine_type = cfg["engine"]["type"]
    if engine_type not in {
        "minicpm_vllm_openai_api",
        "minicpm_sglang_openai_api",
        "minicpm_llamacpp_openai_api",
        "minicpm_ollama_api",
        "minicpm_transformers_local",
    }:
        raise ValueError(f"Unsupported engine type for v0.1: {engine_type}")

    input_cfg = cfg["input"]
    has_any_input = bool(input_cfg.get("image_dir") or input_cfg.get("files_dir") or input_cfg.get("file_path"))
    if not has_any_input:
        raise ValueError("At least one of input.image_dir / input.files_dir / input.file_path is required")

    if input_cfg.get("image_dir") and not Path(input_cfg["image_dir"]).exists():
        raise FileNotFoundError(f"input.image_dir not found: {input_cfg['image_dir']}")
    if input_cfg.get("files_dir") and not Path(input_cfg["files_dir"]).exists():
        raise FileNotFoundError(f"input.files_dir not found: {input_cfg['files_dir']}")
    if input_cfg.get("file_path"):
        file_path = Path(input_cfg["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(f"input.file_path not found: {input_cfg['file_path']}")
        if file_path.suffix.lower() not in ALLOWED_FILE_SUFFIXES:
            raise ValueError(f"Unsupported input.file_path suffix: {file_path.suffix}")

    if not cfg["layout"].get("enabled", True):
        raise ValueError("layout.enabled must be true in SDK core pipeline")
    if cfg["layout"].get("provider", "") != "ppdoclayout3":
        raise ValueError(f"Unsupported layout provider: {cfg['layout'].get('provider')}")
    if not cfg["layout"]["model_dir"]:
        raise ValueError("layout.model_dir is required")
    if not Path(cfg["layout"]["model_dir"]).exists():
        raise FileNotFoundError(f"layout.model_dir not found: {cfg['layout']['model_dir']}")

    if cfg["engine"].get("model_path") and not Path(cfg["engine"]["model_path"]).exists():
        raise FileNotFoundError(f"engine.model_path not found: {cfg['engine']['model_path']}")
    if engine_type == "minicpm_sglang_openai_api" and cfg["engine"].get("auto_start", True):
        if not cfg["engine"].get("model_path"):
            raise ValueError("engine.model_path is required for the SGLang backend")
        if not cfg["engine"].get("sglang_python"):
            raise ValueError(
                "engine.sglang_python or OCRCPM_SGLANG_PYTHON is required "
                "when SGLang auto_start=true"
            )
        sglang_python = str(cfg["engine"]["sglang_python"])
        if ("/" in sglang_python or "\\" in sglang_python) and not Path(sglang_python).exists():
            raise FileNotFoundError(f"engine.sglang_python not found: {sglang_python}")
    if engine_type == "minicpm_llamacpp_openai_api" and cfg["engine"].get("auto_start", True):
        mmproj_path = cfg["engine"].get("mmproj_path", "")
        if not mmproj_path:
            raise ValueError("engine.mmproj_path is required for the llama.cpp backend")
        if not Path(mmproj_path).exists():
            raise FileNotFoundError(f"engine.mmproj_path not found: {mmproj_path}")
        if not cfg["engine"].get("llama_server"):
            raise ValueError(
                "engine.llama_server or OCRCPM_LLAMA_SERVER is required "
                "when llama.cpp auto_start=true"
            )
        llama_server = str(cfg["engine"]["llama_server"])
        if ("/" in llama_server or "\\" in llama_server) and not Path(llama_server).exists():
            raise FileNotFoundError(f"engine.llama_server not found: {llama_server}")
    if engine_type == "minicpm_ollama_api":
        model_path = cfg["engine"].get("model_path", "")
        mmproj_path = cfg["engine"].get("mmproj_path", "")
        if bool(model_path) != bool(mmproj_path):
            raise ValueError(
                "engine.model_path and engine.mmproj_path must both be set, so OCRCPM can "
                "register the model in Ollama, or both be empty to use a model that is "
                "already registered"
            )
        if mmproj_path and not Path(mmproj_path).exists():
            raise FileNotFoundError(f"engine.mmproj_path not found: {mmproj_path}")
        ollama_bin = str(cfg["engine"].get("ollama_bin", ""))
        if ("/" in ollama_bin or "\\" in ollama_bin) and not Path(ollama_bin).exists():
            raise FileNotFoundError(f"engine.ollama_bin not found: {ollama_bin}")

    server_urls = cfg["engine"].get("server_urls", [])
    if server_urls is None:
        server_urls = []
    if not isinstance(server_urls, list):
        raise ValueError("engine.server_urls must be a list of URLs")
    for u in server_urls:
        if not isinstance(u, str) or not u.strip():
            raise ValueError("engine.server_urls contains invalid URL value")
    cfg["engine"]["server_urls"] = [u.strip() for u in server_urls if u and u.strip()]

    if int(cfg["pipeline"].get("max_workers", 1) or 1) <= 0:
        raise ValueError("pipeline.max_workers must be >= 1")
    if int(cfg["pipeline"].get("page_workers", 1) or 1) <= 0:
        raise ValueError("pipeline.page_workers must be >= 1")



def load_config(config_path: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _deep_merge(raw, overrides)

    cfg = _deep_merge(DEFAULT_CONFIG, raw)
    base_dir = path.parent

    if not cfg["run"].get("run_tag"):
        cfg["run"]["run_tag"] = f"ocrcpm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cfg["run"]["run_root"] = _resolve_path(base_dir, cfg["run"]["run_root"])

    cfg["input"]["manifest"] = _resolve_path(base_dir, cfg["input"].get("manifest", ""))
    cfg["input"]["file_path"] = _resolve_path(base_dir, cfg["input"].get("file_path", ""))
    cfg["input"]["image_dir"] = _resolve_path(base_dir, cfg["input"].get("image_dir", ""))
    cfg["input"]["files_dir"] = _resolve_path(base_dir, cfg["input"].get("files_dir", ""))

    cfg["prompt"]["prompt_map"] = _resolve_path(base_dir, cfg["prompt"].get("prompt_map", ""))
    cfg["engine"]["model_path"] = _resolve_path(base_dir, cfg["engine"].get("model_path", ""))
    cfg["engine"]["mmproj_path"] = _resolve_path(base_dir, cfg["engine"].get("mmproj_path", ""))
    cfg["layout"]["model_dir"] = _resolve_path(base_dir, cfg["layout"].get("model_dir", ""))

    # Allow environment overrides so machine-specific interpreter/toolkit paths
    # do not have to be baked into configs (MinerU-style pluggable backend setup).
    if not cfg["engine"].get("vllm_python"):
        cfg["engine"]["vllm_python"] = os.environ.get("OCRCPM_VLLM_PYTHON", "")
    if not cfg["engine"].get("sglang_python"):
        cfg["engine"]["sglang_python"] = os.environ.get("OCRCPM_SGLANG_PYTHON", "")
    if not cfg["engine"].get("llama_server"):
        cfg["engine"]["llama_server"] = os.environ.get("OCRCPM_LLAMA_SERVER", "")
    llama_server = str(cfg["engine"].get("llama_server", ""))
    if llama_server and ("/" in llama_server or "\\" in llama_server):
        cfg["engine"]["llama_server"] = _resolve_path(base_dir, llama_server)
    if not cfg["engine"].get("ollama_bin"):
        cfg["engine"]["ollama_bin"] = os.environ.get("OCRCPM_OLLAMA_BIN", "")
    ollama_bin = str(cfg["engine"].get("ollama_bin", ""))
    if ollama_bin and ("/" in ollama_bin or "\\" in ollama_bin):
        cfg["engine"]["ollama_bin"] = _resolve_path(base_dir, ollama_bin)
    if not cfg["engine"].get("cuda_home"):
        cfg["engine"]["cuda_home"] = os.environ.get("OCRCPM_CUDA_HOME", "")

    run_dir = Path(cfg["run"]["run_root"]) / cfg["run"]["run_tag"]
    runtime_intermediate_dir = run_dir / "runtime" / "intermediate"
    cfg["_meta"] = {
        "config_path": str(path),
        "run_dir": str(run_dir),
        "runtime_dir": str(run_dir / "runtime"),
        "runtime_intermediate_dir": str(runtime_intermediate_dir),
        "output_dir": str(run_dir / "output"),
        "result_json_dir": str(run_dir / "results_json"),
        "log_dir": str(run_dir / "logs"),
        "active_intermediate_dir": str(runtime_intermediate_dir),
    }

    _validate_config(cfg)
    return cfg
