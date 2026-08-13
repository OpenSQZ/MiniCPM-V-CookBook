from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import yaml


DEFAULT_LAYOUT_REPO = "PaddlePaddle/PP-DocLayoutV3_safetensors"
DEFAULT_MODEL_ROOT = Path(os.environ.get("OCRCPM_MODEL_DIR", "~/.cache/ocrcpm/models")).expanduser()
DEFAULT_CONFIG_PATH = Path(os.environ.get("OCRCPM_CONFIG", "~/.config/ocrcpm/config.yaml")).expanduser()


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _repo_leaf(repo_id: str, fallback: str) -> str:
    leaf = repo_id.rstrip("/").split("/")[-1]
    return leaf or fallback


def _download_huggingface(repo_id: str, local_dir: Path, revision: str = "") -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --source huggingface. "
            "Install it with: pip install 'ocrcpm[models]' or pip install huggingface_hub"
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs: Dict[str, Any] = {"repo_id": repo_id, "local_dir": str(local_dir)}
    if revision:
        kwargs["revision"] = revision
    return _resolve_path(snapshot_download(**kwargs))


def _download_modelscope(repo_id: str, local_dir: Path, revision: str = "") -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "modelscope is required for --source modelscope. "
            "Install it with: pip install 'ocrcpm[models]' or pip install modelscope"
        ) from exc

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    kwargs: Dict[str, Any] = {"model_id": repo_id, "cache_dir": str(local_dir.parent)}
    if revision:
        kwargs["revision"] = revision
    return _resolve_path(snapshot_download(**kwargs))


def _download_snapshot(source: str, repo_id: str, local_dir: Path, revision: str = "") -> Path:
    if source == "huggingface":
        return _download_huggingface(repo_id, local_dir, revision=revision)
    if source == "modelscope":
        return _download_modelscope(repo_id, local_dir, revision=revision)
    raise ValueError(f"Unsupported model source: {source}")


def _server_host_port(server_url: str) -> tuple[str, int]:
    parsed = urlparse(server_url)
    return parsed.hostname or "127.0.0.1", parsed.port or 18599


def _write_config(
    *,
    output_path: Path,
    backend: str,
    layout_dir: Path,
    ocr_dir: Path,
    input_path: str,
    run_root: str,
    layout_device: str,
    server_url: str,
    model_name: str,
    transformers_device: str,
    torch_dtype: str,
    vllm_python: str,
    sglang_python: str,
    sglang_jit_cache_dir: str,
    llama_server: str,
    llama_model: Path | None,
    llama_mmproj: Path | None,
    llama_gpu_layers: int,
    llama_parallel: int,
    ollama_bin: str,
    ollama_model_gguf: Path | None,
    ollama_mmproj: Path | None,
    ollama_models_dir: str,
    cuda_home: str,
    gpu_id: str,
    port: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    image_limit: int,
    gpu_memory_utilization: float,
    max_workers: int,
    page_workers: int,
    max_slice_nums: int,
    downsample_mode: str,
) -> None:
    if backend == "transformers-local":
        engine: Dict[str, Any] = {
            "type": "minicpm_transformers_local",
            "model_path": str(ocr_dir),
            "model_name": model_name,
            "device": transformers_device,
            "torch_dtype": torch_dtype,
            "gpu_id": gpu_id,
            "auto_start": False,
        }
    elif backend == "llama-cpp":
        if llama_model is None or llama_mmproj is None:
            raise ValueError("--llama-model and --llama-mmproj are required for --backend llama-cpp")
        engine = {
            "type": "minicpm_llamacpp_openai_api",
            "model_path": str(llama_model),
            "mmproj_path": str(llama_mmproj),
            "model_name": model_name,
            "llama_server": llama_server,
            "auto_start": True,
            "host": "127.0.0.1",
            "port": port,
            "gpu_id": gpu_id,
            "gpu_layers": llama_gpu_layers,
            "parallel": llama_parallel,
            "max_model_len": max_model_len,
            "startup_timeout_sec": 1800,
            "wait_ready": True,
            "request_timeout_sec": 600,
            "request_ready_timeout_sec": 900,
        }
    elif backend == "ollama":
        engine = {
            "type": "minicpm_ollama_api",
            "model_name": model_name,
            "ollama_bin": ollama_bin,
            "ollama_models_dir": ollama_models_dir,
            "auto_start": True,
            "host": "127.0.0.1",
            "port": port,
            "gpu_id": gpu_id,
            "gpu_layers": llama_gpu_layers,
            "parallel": llama_parallel,
            "keep_alive": "30m",
            "max_model_len": max_model_len,
            "startup_timeout_sec": 1800,
            "wait_ready": True,
            "request_timeout_sec": 600,
            "request_ready_timeout_sec": 900,
        }
        # Both GGUF paths let OCRCPM register the model on first run; without
        # them the tag must already exist in Ollama.
        if ollama_model_gguf is not None and ollama_mmproj is not None:
            engine["model_path"] = str(ollama_model_gguf)
            engine["mmproj_path"] = str(ollama_mmproj)
    elif backend == "sglang":
        engine = {
            "type": "minicpm_sglang_openai_api",
            "model_path": str(ocr_dir),
            "model_name": model_name,
            "sglang_python": sglang_python,
            "sglang_jit_cache_dir": sglang_jit_cache_dir,
            "auto_start": True,
            "host": "127.0.0.1",
            "port": port,
            "gpu_id": gpu_id,
            "torch_dtype": torch_dtype,
            "cuda_home": cuda_home,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "gpu_memory_utilization": gpu_memory_utilization,
            "startup_timeout_sec": 1800,
            "wait_ready": True,
            "request_timeout_sec": 600,
            "request_ready_timeout_sec": 900,
        }
    else:
        engine = {
            "type": "minicpm_vllm_openai_api",
            "model_path": str(ocr_dir),
            "model_name": model_name,
            "request_timeout_sec": 600,
            "request_ready_timeout_sec": 900,
        }

    if backend == "vllm-auto":
        engine.update(
            {
                "auto_start": True,
                "vllm_python": vllm_python,
                "cuda_home": cuda_home,
                "gpu_id": gpu_id,
                "port": port,
                "max_model_len": max_model_len,
                "max_num_batched_tokens": max_num_batched_tokens,
                "image_limit": image_limit,
                "gpu_memory_utilization": gpu_memory_utilization,
                "startup_timeout_sec": 1800,
                "wait_ready": True,
            }
        )
    elif backend in {"hf-server", "http-client"}:
        engine.update({"auto_start": False, "server_urls": [server_url]})

    cfg: Dict[str, Any] = {
        "run": {
            "run_tag": "ocrcpm_example",
            "run_root": run_root,
            "mode": "full",
            "seed": 42,
        },
        "input": {
            "file_path": input_path,
            "limit": 0,
        },
        "layout": {
            "enabled": True,
            "provider": "ppdoclayout3",
            "model_dir": str(layout_dir),
            "device": layout_device,
        },
        "engine": engine,
        "pipeline": {
            "max_workers": max_workers,
            "page_workers": page_workers,
            "queue_mode": "serial",
        },
        "mm": {
            "image_limit": image_limit,
            "max_slice_nums": max_slice_nums,
            "downsample_mode": downsample_mode,
        },
    }
    if backend in {"llama-cpp", "ollama"}:
        cfg["decode"] = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 0,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def download_models_and_write_config(args: Any) -> Dict[str, Any]:
    output_dir = _resolve_path(args.output_dir)
    layout_dir = _resolve_path(args.layout_dir) if args.layout_dir else output_dir / _repo_leaf(args.layout_repo, "layout")
    ocr_dir = _resolve_path(args.ocr_dir) if args.ocr_dir else output_dir / _repo_leaf(args.ocr_repo, "ocr_model")
    llama_model = _resolve_path(args.llama_model) if args.llama_model else None
    llama_mmproj = _resolve_path(args.llama_mmproj) if args.llama_mmproj else None
    ollama_model_gguf = (
        _resolve_path(args.ollama_model_gguf) if getattr(args, "ollama_model_gguf", "") else None
    )
    ollama_mmproj = _resolve_path(args.ollama_mmproj) if getattr(args, "ollama_mmproj", "") else None

    if args.backend == "llama-cpp" and (
        not args.llama_server or llama_model is None or llama_mmproj is None
    ):
        raise ValueError(
            "--llama-server, --llama-model and --llama-mmproj are required "
            "for --backend llama-cpp"
        )
    if args.backend == "ollama" and (ollama_model_gguf is None) != (ollama_mmproj is None):
        raise ValueError(
            "--ollama-model-gguf and --ollama-mmproj must be passed together, or "
            "both omitted to use a model that is already registered in Ollama"
        )
    if args.backend == "sglang" and not getattr(args, "sglang_python", ""):
        raise ValueError(
            "--sglang-python or OCRCPM_SGLANG_PYTHON is required "
            "for --backend sglang"
        )
    if args.backend not in {"llama-cpp", "ollama"} and not args.ocr_repo and not args.ocr_dir and not args.skip_download:
        raise ValueError("OCR model repo is required. Pass --ocr-repo <repo-id> or --ocr-dir <local-model-dir>.")

    downloads: Dict[str, Dict[str, Any]] = {}
    if args.layout_dir:
        downloads["layout"] = {"status": "local", "path": str(layout_dir)}
    elif args.skip_download:
        downloads["layout"] = {"status": "skipped", "path": str(layout_dir), "repo": args.layout_repo}
    else:
        layout_dir = _download_snapshot(args.source, args.layout_repo, layout_dir, revision=args.layout_revision)
        downloads["layout"] = {"status": "downloaded", "path": str(layout_dir), "repo": args.layout_repo}

    if args.backend == "llama-cpp":
        downloads["ocr"] = {
            "status": "local",
            "model_path": str(llama_model),
            "mmproj_path": str(llama_mmproj),
        }
    elif args.backend == "ollama":
        downloads["ocr"] = (
            {
                "status": "local",
                "model_path": str(ollama_model_gguf),
                "mmproj_path": str(ollama_mmproj),
            }
            if ollama_model_gguf is not None
            else {"status": "registered", "ollama_model": args.model_name}
        )
    elif args.ocr_dir:
        downloads["ocr"] = {"status": "local", "path": str(ocr_dir)}
    elif args.skip_download:
        downloads["ocr"] = {"status": "skipped", "path": str(ocr_dir), "repo": args.ocr_repo}
    else:
        ocr_dir = _download_snapshot(args.source, args.ocr_repo, ocr_dir, revision=args.ocr_revision)
        downloads["ocr"] = {"status": "downloaded", "path": str(ocr_dir), "repo": args.ocr_repo}

    config_output = (
        _resolve_path(args.config_output)
        if args.config_output
        else _resolve_path(DEFAULT_CONFIG_PATH if getattr(args, "install_default", False) else output_dir / "ocrcpm_config.yaml")
    )
    _write_config(
        output_path=config_output,
        backend=args.backend,
        layout_dir=layout_dir,
        ocr_dir=ocr_dir,
        input_path=args.input,
        run_root=args.run_root,
        layout_device=args.layout_device,
        server_url=args.server_url,
        model_name=args.model_name,
        transformers_device=args.transformers_device,
        torch_dtype=args.torch_dtype,
        vllm_python=args.vllm_python,
        sglang_python=getattr(args, "sglang_python", ""),
        sglang_jit_cache_dir=getattr(args, "sglang_jit_cache_dir", ""),
        llama_server=args.llama_server,
        llama_model=llama_model,
        llama_mmproj=llama_mmproj,
        llama_gpu_layers=args.llama_gpu_layers,
        llama_parallel=args.llama_parallel,
        ollama_bin=getattr(args, "ollama_bin", ""),
        ollama_model_gguf=ollama_model_gguf,
        ollama_mmproj=ollama_mmproj,
        ollama_models_dir=getattr(args, "ollama_models_dir", ""),
        cuda_home=args.cuda_home,
        gpu_id=args.gpu_id,
        port=args.port,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        image_limit=args.image_limit,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_workers=args.max_workers,
        page_workers=args.page_workers,
        max_slice_nums=args.max_slice_nums,
        downsample_mode=args.downsample_mode,
    )

    summary: Dict[str, Any] = {
        "command": getattr(args, "command", "download-models"),
        "source": args.source,
        "models": downloads,
        "config": str(config_output),
        "backend": args.backend,
        "next_parse_command": (
            "ocrcpm parse --input <input.pdf|png|jpg> --output out.md"
            if getattr(args, "install_default", False)
            else f"ocrcpm parse --config {config_output} --input <input.pdf|png|jpg> --output out.md"
        ),
    }
    if getattr(args, "install_default", False):
        summary["default_config"] = True
    if args.backend == "hf-server":
        host, port = _server_host_port(args.server_url)
        summary["server_command"] = f"ocrcpm-server --model-path {ocr_dir} --host {host} --port {port}"
    elif args.backend == "llama-cpp":
        summary["server_command"] = (
            f"{args.llama_server} -m {llama_model} --mmproj {llama_mmproj} "
            f"-ngl {args.llama_gpu_layers} -c {args.max_model_len} "
            f"-np {args.llama_parallel} --repeat-penalty 1.1 "
            f"--host 127.0.0.1 --port {args.port} --jinja"
        )
    elif args.backend == "ollama":
        summary["server_command"] = (
            f"OLLAMA_HOST=127.0.0.1:{args.port} "
            f"OLLAMA_CONTEXT_LENGTH={args.max_model_len} "
            f"CUDA_VISIBLE_DEVICES={args.gpu_id} "
            f"{getattr(args, 'ollama_bin', '') or 'ollama'} serve"
        )
        if ollama_model_gguf is not None:
            summary["model_create_command"] = (
                f"OLLAMA_HOST=127.0.0.1:{args.port} "
                f"{getattr(args, 'ollama_bin', '') or 'ollama'} create {args.model_name} -f Modelfile"
            )
    elif args.backend == "sglang":
        summary["server_command"] = (
            f"{args.sglang_python} -m sglang.launch_server "
            f"--model-path {ocr_dir} --served-model-name {args.model_name} "
            f"--trust-remote-code --dtype {args.torch_dtype} "
            f"--context-length {args.max_model_len} "
            f"--mem-fraction-static {args.gpu_memory_utilization} "
            f"--json-model-override-args "
            f"'{{\"downsample_mode\":\"{args.downsample_mode}\"}}' "
            f"--host 127.0.0.1 --port {args.port}"
        )
    return summary
