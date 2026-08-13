from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from .config import load_config
from .model_assets import DEFAULT_CONFIG_PATH, DEFAULT_LAYOUT_REPO, DEFAULT_MODEL_ROOT, download_models_and_write_config
from .pipeline import infer_run, parse_layout_run, prepare_run, read_markdown_from_run


def _add_model_config_args(p: argparse.ArgumentParser, *, default_backend: str, config_output_help: str) -> None:
    p.add_argument("--source", choices=("huggingface", "modelscope"), default=os.environ.get("OCRCPM_MODEL_SOURCE", "huggingface"))
    p.add_argument("--output-dir", default=str(DEFAULT_MODEL_ROOT), help="Directory for downloaded model snapshots")
    p.add_argument("--layout-repo", default=os.environ.get("OCRCPM_LAYOUT_REPO", DEFAULT_LAYOUT_REPO))
    p.add_argument("--ocr-repo", default=os.environ.get("OCRCPM_OCR_REPO", ""), help="OCR model repo id; required unless --ocr-dir or --skip-download is used")
    p.add_argument("--layout-revision", default="")
    p.add_argument("--ocr-revision", default="")
    p.add_argument("--layout-dir", default="", help="Use an existing local layout model directory instead of downloading it")
    p.add_argument("--ocr-dir", default="", help="Use an existing local OCR model directory instead of downloading it")
    p.add_argument("--skip-download", action="store_true", help="Only write the config; do not download model files")
    p.add_argument("--config-output", default="", help=config_output_help)
    p.add_argument(
        "--backend",
        choices=(
            "transformers-local",
            "hf-server",
            "http-client",
            "vllm-auto",
            "sglang",
            "llama-cpp",
            "ollama",
        ),
        default=default_backend,
    )
    p.add_argument("--server-url", default="http://127.0.0.1:18599", help="OpenAI-compatible endpoint for hf-server/http-client configs")
    p.add_argument("--input", default="/abs/path/to/input.pdf", help="Input placeholder written into the generated config")
    p.add_argument("--run-root", default="./runs")
    p.add_argument("--layout-device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--model-name", default="minicpm-v4.7")
    p.add_argument("--transformers-device", default="cuda", choices=("cpu", "cuda"), help="Device for --backend transformers-local")
    p.add_argument("--torch-dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"), help="Model dtype for --backend transformers-local")
    p.add_argument("--vllm-python", default=os.environ.get("OCRCPM_VLLM_PYTHON", ""))
    p.add_argument(
        "--sglang-python",
        default=os.environ.get("OCRCPM_SGLANG_PYTHON", ""),
        help="Path to the SGLang environment's Python for --backend sglang",
    )
    p.add_argument(
        "--sglang-jit-cache-dir",
        default="",
        help="Optional isolated TVM-FFI JIT cache for --backend sglang",
    )
    p.add_argument(
        "--llama-server",
        default=os.environ.get("OCRCPM_LLAMA_SERVER", ""),
        help="Path to llama-server for --backend llama-cpp",
    )
    p.add_argument("--llama-model", default="", help="Language-model GGUF for --backend llama-cpp")
    p.add_argument("--llama-mmproj", default="", help="Multimodal-projector GGUF for --backend llama-cpp")
    p.add_argument("--llama-gpu-layers", type=int, default=99, help="GPU layers for the llama-cpp and ollama backends")
    p.add_argument("--llama-parallel", type=int, default=1, help="Parallel slots for the llama-cpp and ollama backends")
    p.add_argument(
        "--ollama-bin",
        default=os.environ.get("OCRCPM_OLLAMA_BIN", ""),
        help="Path to the ollama binary for --backend ollama; defaults to `ollama` on PATH",
    )
    p.add_argument(
        "--ollama-model-gguf",
        default="",
        help="Language-model GGUF registered into Ollama on first run for --backend ollama",
    )
    p.add_argument(
        "--ollama-mmproj",
        default="",
        help="Multimodal-projector GGUF registered into Ollama on first run for --backend ollama",
    )
    p.add_argument(
        "--ollama-models-dir",
        default="",
        help="Override the Ollama blob store directory (OLLAMA_MODELS) for --backend ollama",
    )
    p.add_argument("--cuda-home", default=os.environ.get("OCRCPM_CUDA_HOME", ""))
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--port", type=int, default=18499)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--page-workers", type=int, default=1)
    p.add_argument("--image-limit", type=int, default=1)
    p.add_argument("--max-slice-nums", type=int, default=9)
    p.add_argument("--downsample-mode", default="4x")


def _resolve_config_path(config_arg: str) -> str:
    if config_arg:
        return config_arg
    default_config = DEFAULT_CONFIG_PATH.expanduser()
    if default_config.exists():
        return str(default_config)
    raise FileNotFoundError(
        "No config provided and default config not found: {}. "
        "Run `ocrcpm init ...` once, or pass --config <path>.".format(default_config)
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocrcpm", description="ocrcpm: MiniCPM-V OCR pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("parse-layout", "prepare", "infer", "run", "show-config"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="", help=f"Path to pipeline YAML config; defaults to {DEFAULT_CONFIG_PATH}")

    p_parse = sub.add_parser("parse", help="One-shot parse: file path -> markdown")
    p_parse.add_argument("--config", default="", help=f"Path to pipeline YAML config; defaults to {DEFAULT_CONFIG_PATH}")
    p_parse.add_argument("--input", required=True, help="Single file path: pdf/png/jpg/jpeg")
    p_parse.add_argument("--output", default="", help="Output markdown path; default is stdout")
    p_parse.add_argument("--stdout", action="store_true", help="Print markdown to stdout")
    p_parse.add_argument("--run-tag", default="", help="Optional run tag for this one-shot run")
    p_parse.add_argument("--no-merge-pages", action="store_true", help="Do not merge pages into a single markdown")
    p_parse.add_argument("--page-delimiter", default="\n\n<!-- page-break -->\n\n", help="Delimiter between pages when merging")

    p_models = sub.add_parser("download-models", help="Download model weights and write a ready-to-edit config")
    _add_model_config_args(
        p_models,
        default_backend="transformers-local",
        config_output_help="Path for generated YAML config; defaults to <output-dir>/ocrcpm_config.yaml",
    )

    p_init = sub.add_parser("init", help="Initialize the default OCRCPM config used by parse/run")
    _add_model_config_args(
        p_init,
        default_backend="transformers-local",
        config_output_help=f"Path for generated YAML config; defaults to {DEFAULT_CONFIG_PATH}",
    )
    p_init.set_defaults(install_default=True)

    return parser


def _build_parse_overrides(args: argparse.Namespace) -> dict:
    input_path = Path(args.input).resolve()
    run_tag = args.run_tag or f"ocrcpm_parse_{input_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return {
        "run": {
            "run_tag": run_tag,
            "mode": "oneshot_file",
        },
        "input": {
            "file_path": str(input_path),
            "manifest": "",
            "image_dir": "",
            "files_dir": "",
            "limit": 0,
        },
        "layout": {
            "enabled": True,
        },
        "logging": {
            # Keep one-shot stdout clean by suppressing infer progress lines.
            "progress_log_every": 10**9,
        },
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in {"download-models", "init"}:
        summary = download_models_and_write_config(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "parse":
        overrides = _build_parse_overrides(args)
        try:
            config_path = _resolve_config_path(args.config)
        except FileNotFoundError as e:
            parser.error(str(e))
        cfg = load_config(config_path, overrides=overrides)

        layout_summary = parse_layout_run(cfg)
        prepare_run(cfg)
        infer_summary = infer_run(cfg)

        run_dir = Path(cfg["_meta"]["run_dir"])
        markdown, pages = read_markdown_from_run(
            run_dir=run_dir,
            merge_pages=not args.no_merge_pages,
            page_delimiter=args.page_delimiter,
        )

        if args.stdout or not args.output:
            print(markdown)
            return 0

        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

        print(
            json.dumps(
                {
                    "command": "parse",
                    "input": str(Path(args.input).resolve()),
                    "output_md": str(output_path),
                    "pages": pages,
                    "run_dir": str(run_dir),
                    "layout_summary": layout_summary,
                    "infer_summary": infer_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        config_path = _resolve_config_path(args.config)
    except FileNotFoundError as e:
        parser.error(str(e))
    cfg = load_config(config_path)

    if args.command == "show-config":
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return 0

    if args.command == "parse-layout":
        summary = parse_layout_run(cfg)
        print(json.dumps({"command": "parse-layout", "run_dir": cfg["_meta"]["run_dir"], "summary": summary}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "prepare":
        meta = prepare_run(cfg)
        print(json.dumps({"command": "prepare", "run_dir": cfg["_meta"]["run_dir"], "planned_pages": meta["counts"]["planned_pages"]}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "infer":
        summary = infer_run(cfg)
        print(json.dumps({"command": "infer", "run_dir": cfg["_meta"]["run_dir"], "summary": summary}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run":
        layout_summary = parse_layout_run(cfg)
        prepare_run(cfg)
        infer_summary = infer_run(cfg)
        print(
            json.dumps(
                {
                    "command": "run",
                    "run_dir": cfg["_meta"]["run_dir"],
                    "layout_summary": layout_summary,
                    "infer_summary": infer_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
