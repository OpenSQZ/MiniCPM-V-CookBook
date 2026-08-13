from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from PIL import Image
import yaml

from .default_prompts import DEFAULT_PROMPT_MAP, apply_prompt_override_mode
from .engine import create_engine
from .markdown_utils import assemble_markdown, postprocess_content
from .metrics import build_summary, write_per_page_metrics, write_summary
from .utils import append_jsonl, ensure_dir, now_iso, read_json, read_jsonl, write_json

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SUPPORTED_INPUT_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}
LABEL_MAP = {"formula": "display_formula"}
INLINE_LATEX_RE = re.compile(r"\\\((.+?)\\\)")
DISPLAY_BRACKET_SEP_RE = re.compile(re.escape(r"\]") + r"\s*" + re.escape(r"\["))


def _load_prompt_map(cfg: Dict) -> Dict:
    prompt_map_path = cfg["prompt"].get("prompt_map", "")
    prompt_map = json.loads(json.dumps(DEFAULT_PROMPT_MAP))
    if prompt_map_path and Path(prompt_map_path).is_file():
        with Path(prompt_map_path).open("r", encoding="utf-8") as f:
            external = yaml.safe_load(f) or {}
        prompt_map.update(external)
    return apply_prompt_override_mode(prompt_map, cfg["prompt"].get("override_mode", "legacy"))


def _get_element_cfg(prompt_map: Dict, label: str) -> Dict | None:
    default = prompt_map.get("default", {})
    elem = prompt_map.get("element_prompts", {}).get(label, {})
    if elem.get("skip", False):
        return None
    return {
        "prompt": elem.get("prompt", default.get("prompt", "Text Recognition:")),
        "max_new_tokens": elem.get("max_new_tokens", default.get("max_new_tokens", 4096)),
        "output_format": elem.get("output_format", default.get("output_format", "text")),
        "resize_min_side": elem.get("resize_min_side", default.get("resize_min_side", None)),
        "slice_nums": elem.get("slice_nums", default.get("slice_nums", 9)),
    }


def _resize_image(image: Image.Image, target_min_side) -> Image.Image:
    if target_min_side is None:
        return image
    w, h = image.size
    if min(w, h) >= target_min_side:
        return image
    scale = target_min_side / min(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _mode_limit(cfg: Dict) -> int:
    input_limit = int(cfg["input"].get("limit") or 0)
    if input_limit > 0:
        return input_limit
    mode = cfg["run"]["mode"]
    if mode == "smoke_1page":
        return 1
    if mode == "pipeline_30pages":
        return 30
    return 0


def read_markdown_from_run(
    run_dir: Path,
    *,
    merge_pages: bool = True,
    page_delimiter: str = "\n\n<!-- page-break -->\n\n",
) -> tuple[str, int]:
    tasks = read_jsonl(run_dir / "runtime" / "page_tasks.jsonl")
    if not tasks:
        raise RuntimeError("No page tasks found in run. parse-layout/prepare may have failed.")

    md_root = run_dir / "output"
    page_texts = []
    missing = []
    for task in tasks:
        image_id = task["image_id"]
        md_path = md_root / f"{image_id}.md"
        if not md_path.exists():
            missing.append(str(md_path))
            continue
        page_texts.append(md_path.read_text(encoding="utf-8"))

    if missing:
        raise RuntimeError("Missing markdown outputs: %s" % ", ".join(missing[:5]))

    if not merge_pages and len(page_texts) > 1:
        raise RuntimeError("Multiple pages generated; enable page merging or parse a single-page input.")

    if merge_pages:
        return page_delimiter.join(page_texts), len(page_texts)
    return page_texts[0], 1


def _update_run_meta(run_meta_path: Path, updates: Dict) -> None:
    meta = read_json(run_meta_path, default={}) or {}
    meta.update(updates)
    write_json(run_meta_path, meta)


def _truncate_repetitive_suffix(text: str) -> str:
    words = list(re.finditer(r"\S+", text))
    if len(words) < 18:
        return text

    normalized = [match.group(0).lower() for match in words]
    earliest_cut = len(text)
    max_window = min(64, len(words) // 3)
    for window in range(6, max_window + 1):
        limit = len(words) - (window * 3) + 1
        for start in range(limit):
            first = normalized[start : start + window]
            if (
                first == normalized[start + window : start + (window * 2)]
                and first == normalized[start + (window * 2) : start + (window * 3)]
            ):
                earliest_cut = min(earliest_cut, words[start].start())
                break

    if earliest_cut < len(text):
        return text[:earliest_cut].rstrip()
    return text


def _normalize_crop_text(text: str, label: str) -> str:
    if text is None:
        return ""
    out = _truncate_repetitive_suffix(text.strip())
    if label != "display_formula":
        out = INLINE_LATEX_RE.sub(lambda m: f"${m.group(1).strip()}$", out)
    else:
        out = DISPLAY_BRACKET_SEP_RE.sub("$$   $$", out)
        out = out.replace("\\[", "").replace("\\]", "")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def _normalize_and_dedupe_blocks(blocks: List[Dict]) -> List[Dict]:
    out = []
    seen = set()
    for block in sorted(blocks, key=lambda x: x.get("order", 0)):
        item = dict(block)
        item["content"] = _normalize_crop_text(item.get("content", ""), item.get("label", ""))
        key = (item.get("order"), item.get("label"), item.get("content"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _load_manifest_filter(manifest_path: str) -> set[str]:
    if not manifest_path:
        return set()
    names = set()
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            val = line.strip()
            if not val or val.startswith("#"):
                continue
            names.add(Path(val).stem)
            names.add(Path(val).name)
            names.add(val)
    return names


def _match_manifest(path: Path, selected: set[str]) -> bool:
    if not selected:
        return True
    return path.stem in selected or path.name in selected or str(path) in selected


def _collect_raw_inputs(cfg: Dict) -> List[Path]:
    selected = _load_manifest_filter(cfg["input"].get("manifest", ""))
    paths: List[Path] = []
    seen = set()

    file_path = cfg["input"].get("file_path", "")
    if file_path:
        p = Path(file_path)
        if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_SUFFIXES:
            rp = str(p.resolve())
            seen.add(rp)
            paths.append(p)

    image_dir = cfg["input"].get("image_dir", "")
    if image_dir:
        root = Path(image_dir)
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            rp = str(p.resolve())
            if rp in seen:
                continue
            if not _match_manifest(p, selected):
                continue
            seen.add(rp)
            paths.append(p)

    files_dir = cfg["input"].get("files_dir", "")
    if files_dir:
        root = Path(files_dir)
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
                continue
            rp = str(p.resolve())
            if rp in seen:
                continue
            if not _match_manifest(p, selected):
                continue
            seen.add(rp)
            paths.append(p)

    return paths


def _to_polygon_points(raw_points) -> List[List[int]]:
    out = []
    for point in raw_points:
        if isinstance(point, (list, tuple)):
            x, y = point[0], point[1]
        else:
            x, y = point.tolist()
        out.append([int(x), int(y)])
    return out


def _build_layout_from_prediction(
    *,
    image: Image.Image,
    image_id: str,
    source_path: str,
    source_type: str,
    page_index: int | None,
    result,
    model,
    crops_root: Path,
) -> tuple[Dict, Dict]:
    crop_dir = crops_root / image_id
    ensure_dir(crop_dir)

    blocks = []
    raw_boxes = []

    for idx, (score_t, label_t, box_t, poly_t) in enumerate(zip(result["scores"], result["labels"], result["boxes"], result["polygon_points"])):
        score = float(score_t.item())
        raw_label = model.config.id2label[int(label_t.item())]
        label = LABEL_MAP.get(raw_label, raw_label)
        box = [float(v) for v in box_t.tolist()]

        x1 = max(0, min(image.width - 1, int(round(box[0]))))
        y1 = max(0, min(image.height - 1, int(round(box[1]))))
        x2 = max(x1 + 1, min(image.width, int(round(box[2]))))
        y2 = max(y1 + 1, min(image.height, int(round(box[3]))))

        crop_filename = f"{idx:03d}_{label}.png"
        crop_path = crop_dir / crop_filename
        image.crop((x1, y1, x2, y2)).save(crop_path)

        blocks.append(
            {
                "block_id": idx,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "order": idx,
                "crop_filename": crop_filename,
                "crop_width": x2 - x1,
                "crop_height": y2 - y1,
                "skip_vlm": False,
            }
        )

        raw_boxes.append(
            {
                "order": idx + 1,
                "cls_id": int(label_t.item()),
                "label": raw_label,
                "score": round(score, 6),
                "coordinate": [round(v, 2) for v in box],
                "polygon_points": _to_polygon_points(poly_t),
            }
        )

    layout = {
        "image_id": image_id,
        "image_width": image.width,
        "image_height": image.height,
        "num_blocks": len(blocks),
        "blocks": blocks,
        "source": {
            "type": source_type,
            "path": source_path,
            "relative_path": Path(source_path).name,
            "raw_image_id": image_id,
            "original_stem": Path(source_path).stem,
            "page_index": page_index,
        },
    }

    raw = {
        "input_path": source_path,
        "page_index": page_index,
        "boxes": raw_boxes,
    }
    return layout, raw


def _iter_source_pages(cfg: Dict, runtime_intermediate: Path) -> List[Dict]:
    try:
        import fitz
    except Exception:
        fitz = None

    source_files = _collect_raw_inputs(cfg)
    page_entries: List[Dict] = []
    page_images_dir = runtime_intermediate / "page_images"
    ensure_dir(page_images_dir)

    for path in source_files:
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_SUFFIXES:
            page_entries.append(
                {
                    "image_id": path.stem,
                    "source_type": "image",
                    "source_path": str(path),
                    "page_index": None,
                    "image_path": str(path),
                }
            )
            continue

        if suffix == ".pdf":
            if fitz is None:
                raise RuntimeError("PDF input requires PyMuPDF (fitz); install it in current environment.")
            doc = fitz.open(path)
            try:
                for idx in range(doc.page_count):
                    page = doc.load_page(idx)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                    image_id = f"{path.stem}_page_{idx + 1:03d}"
                    rendered_path = page_images_dir / f"{image_id}.png"
                    Image.frombytes("RGB", [pix.width, pix.height], pix.samples).save(rendered_path)
                    page_entries.append(
                        {
                            "image_id": image_id,
                            "source_type": "pdf",
                            "source_path": str(path),
                            "page_index": idx,
                            "image_path": str(rendered_path),
                        }
                    )
            finally:
                doc.close()
            continue

    limit = _mode_limit(cfg)
    if limit > 0:
        page_entries = page_entries[:limit]

    used = set()
    for item in page_entries:
        base = item["image_id"]
        name = base
        seq = 2
        while name in used:
            name = f"{base}_{seq}"
            seq += 1
        used.add(name)
        item["image_id"] = name

    return page_entries


def parse_layout_run(cfg: Dict) -> Dict:
    run_dir = Path(cfg["_meta"]["run_dir"])
    runtime_dir = run_dir / "runtime"
    runtime_intermediate = Path(cfg["_meta"]["runtime_intermediate_dir"])
    layouts_dir = runtime_intermediate / "layouts"
    crops_root = runtime_intermediate / "crops"

    ensure_dir(run_dir)
    ensure_dir(runtime_dir)
    ensure_dir(layouts_dir)
    ensure_dir(crops_root)
    ensure_dir(run_dir / "logs")

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
    except Exception as e:
        raise RuntimeError(f"Failed to import layout parser dependencies (torch/transformers): {e}") from e

    model_dir = cfg["layout"]["model_dir"]
    device_cfg = str(cfg["layout"].get("device", "cpu")).lower()
    device = "cpu"
    if device_cfg.startswith("cuda") and torch.cuda.is_available():
        device = device_cfg

    t0 = time.time()
    model = AutoModelForObjectDetection.from_pretrained(model_dir)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    model_load_sec = time.time() - t0

    page_entries = _iter_source_pages(cfg, runtime_intermediate)
    if not page_entries:
        raise ValueError("No input pages found. Provide input.file_path or input.image_dir or input.files_dir with png/jpg/pdf files.")

    parse_rows = []
    errors = []
    raw_result_dir = runtime_intermediate / "layout_raw"
    ensure_dir(raw_result_dir)

    for entry in page_entries:
        image_id = entry["image_id"]
        page_t0 = time.time()
        try:
            image = Image.open(entry["image_path"]).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            result = processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]])[0]

            layout, raw = _build_layout_from_prediction(
                image=image,
                image_id=image_id,
                source_path=entry["source_path"],
                source_type=entry["source_type"],
                page_index=entry["page_index"],
                result=result,
                model=model,
                crops_root=crops_root,
            )
            write_json(layouts_dir / f"{image_id}.json", layout)
            write_json(raw_result_dir / f"{image_id}.json", raw)
            image.close()

            parse_rows.append(
                {
                    "image_id": image_id,
                    "status": "ok",
                    "source_path": entry["source_path"],
                    "page_index": entry["page_index"],
                    "num_blocks": layout["num_blocks"],
                    "duration_sec": round(time.time() - page_t0, 6),
                }
            )
        except Exception as e:
            errors.append({"image_id": image_id, "error": str(e)})
            parse_rows.append(
                {
                    "image_id": image_id,
                    "status": "failed",
                    "source_path": entry["source_path"],
                    "page_index": entry["page_index"],
                    "num_blocks": 0,
                    "duration_sec": round(time.time() - page_t0, 6),
                    "error": str(e),
                }
            )

    layout_timing_path = runtime_dir / "layout_timing.jsonl"
    with layout_timing_path.open("w", encoding="utf-8") as f:
        for row in parse_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cfg["_meta"]["active_intermediate_dir"] = str(runtime_intermediate)

    summary = {
        "status": "ok" if not errors else "partial_failed",
        "provider": "ppdoclayout3",
        "device": device,
        "model_dir": model_dir,
        "active_intermediate_dir": str(runtime_intermediate),
        "num_pages": len(page_entries),
        "ok_pages": sum(1 for x in parse_rows if x["status"] == "ok"),
        "failed_pages": sum(1 for x in parse_rows if x["status"] != "ok"),
        "model_load_sec": round(model_load_sec, 6),
    }
    if errors:
        summary["errors"] = errors[:20]

    _update_run_meta(
        run_dir / "run_meta.json",
        {
            "parse_layout_finished_at": now_iso(),
            "layout": cfg["layout"],
            "parse_layout_summary": summary,
            "active_intermediate_dir": str(runtime_intermediate),
        },
    )
    return summary


def prepare_run(cfg: Dict) -> Dict:
    run_dir = Path(cfg["_meta"]["run_dir"])
    runtime_dir = run_dir / "runtime"
    ensure_dir(run_dir)
    ensure_dir(runtime_dir)
    ensure_dir(run_dir / "output")
    ensure_dir(run_dir / "results_json")
    ensure_dir(run_dir / "logs")

    active_intermediate = cfg["_meta"].get("active_intermediate_dir")
    if not active_intermediate:
        raise ValueError("No runtime intermediate dir available in config metadata")

    layouts_dir = Path(active_intermediate) / "layouts"
    if not layouts_dir.exists() or not list(layouts_dir.glob("*.json")):
        parse_layout_run(cfg)
        active_intermediate = cfg["_meta"].get("active_intermediate_dir")
        layouts_dir = Path(active_intermediate) / "layouts"

    if not layouts_dir.exists():
        raise FileNotFoundError(f"Layout dir not found: {layouts_dir}")

    layout_files = sorted(layouts_dir.glob("*.json"))
    selected = _load_manifest_filter(cfg["input"].get("manifest", ""))
    if selected:
        layout_files = [p for p in layout_files if p.stem in selected or p.name in selected]

    limit = _mode_limit(cfg)
    if limit > 0:
        layout_files = layout_files[:limit]

    tasks = []
    for layout_path in layout_files:
        image_id = layout_path.stem
        tasks.append(
            {
                "image_id": image_id,
                "layout_path": str(layout_path),
                "crop_dir": str(Path(active_intermediate) / "crops" / image_id),
                "source_path": "",
            }
        )

    tasks_path = runtime_dir / "page_tasks.jsonl"
    with tasks_path.open("w", encoding="utf-8") as f:
        for row in tasks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metric_definition = {
        "pipeline_mode": cfg["run"]["mode"],
        "page_metric": ["page_total_wall_sec", "layout_stage_sec", "extract_prep_sec", "extract_infer_sec", "post_process_sec"],
        "crop_metric": ["label", "duration_sec", "status"],
        "ae_metric": ["image_encode_time_sec", "prefill_time_sec", "decode_time_sec", "decode_speed_tps", "output_token_count"],
    }

    prior_meta = read_json(run_dir / "run_meta.json", default={}) or {}
    run_meta = {
        "run_tag": cfg["run"]["run_tag"],
        "run_root": cfg["run"]["run_root"],
        "run_dir": str(run_dir),
        "created_at": prior_meta.get("created_at", now_iso()),
        "mode": cfg["run"]["mode"],
        "seed": cfg["run"]["seed"],
        "engine": cfg["engine"],
        "layout": cfg["layout"],
        "pipeline": cfg["pipeline"],
        "mm": cfg["mm"],
        "decode": cfg["decode"],
        "prompt": cfg["prompt"],
        "input": cfg["input"],
        "active_intermediate_dir": str(active_intermediate),
        "counts": {"planned_pages": len(tasks)},
        "metric_definition": metric_definition,
    }
    if "parse_layout_summary" in prior_meta:
        run_meta["parse_layout_summary"] = prior_meta["parse_layout_summary"]
    if "parse_layout_finished_at" in prior_meta:
        run_meta["parse_layout_finished_at"] = prior_meta["parse_layout_finished_at"]

    write_json(run_dir / "run_meta.json", run_meta)
    return run_meta


def infer_run(cfg: Dict) -> Dict:
    run_dir = Path(cfg["_meta"]["run_dir"])
    run_meta_path = run_dir / "run_meta.json"
    if not run_meta_path.exists():
        prepare_run(cfg)

    prompt_map = _load_prompt_map(cfg)
    tasks = read_jsonl(run_dir / "runtime" / "page_tasks.jsonl")
    engine = create_engine(cfg, run_dir)
    engine.start()

    per_page: List[Dict] = []
    progress_every = max(1, int(cfg["logging"].get("progress_log_every", 20)))
    crop_timing_path = run_dir / "runtime" / "crop_timing.jsonl"
    page_timing_path = run_dir / "runtime" / "page_timing.jsonl"
    page_workers = max(1, int(cfg["pipeline"].get("page_workers", 1) or 1))

    try:
        def _process_page(task: Dict) -> Dict:
            image_id = task["image_id"]
            page_started = time.time()
            layout_stage_sec = 0.0
            extract_prep_sec = 0.0
            extract_infer_sec = 0.0
            post_process_sec = 0.0
            num_failed_blocks = 0
            num_vlm_blocks = 0
            crop_rows: List[Dict] = []

            layout_path = Path(task["layout_path"])
            crop_dir = Path(task["crop_dir"])
            md_path = run_dir / "output" / f"{image_id}.md"
            json_path = run_dir / "results_json" / f"{image_id}.json"

            try:
                layout_t0 = time.time()
                layout = read_json(layout_path, default={}) or {}
                blocks = layout.get("blocks", [])
                layout_stage_sec = time.time() - layout_t0

                if md_path.exists() and json_path.exists():
                    row = {
                        "image_id": image_id,
                        "status": "ok",
                        "page_total_wall_sec": round(time.time() - page_started, 6),
                        "layout_stage_sec": round(layout_stage_sec, 6),
                        "extract_prep_sec": 0.0,
                        "extract_infer_sec": 0.0,
                        "post_process_sec": 0.0,
                        "num_blocks": len(blocks),
                        "num_vlm_blocks": 0,
                        "num_failed_blocks": 0,
                        "output_chars": len(md_path.read_text(encoding="utf-8")),
                        "error": "",
                    }
                    page_timing = {
                        "image_id": image_id,
                        "status": "skipped_existing",
                        "duration_sec": row["page_total_wall_sec"],
                    }
                    return {"page_row": row, "page_timing": page_timing, "crop_rows": crop_rows}

                result_blocks = []
                block_workers = max(1, int(cfg["pipeline"].get("max_workers", 1) or 1))

                def _run_block(block: Dict) -> Dict:
                    label = block.get("label", "")
                    elem_cfg = _get_element_cfg(prompt_map, label)
                    crop_filename = block.get("crop_filename", "")
                    crop_path = crop_dir / crop_filename if crop_filename else None
                    crop_started = time.time()

                    if elem_cfg is None or block.get("skip_vlm", False):
                        return {
                            "result_block": {
                                "label": label,
                                "order": block.get("order", 0),
                                "bbox": block.get("bbox"),
                                "crop_filename": crop_filename,
                                "status": "skipped_no_vlm",
                                "content": "",
                                "raw_content": "",
                            },
                            "crop_row": {
                                "image_id": image_id,
                                "label": label,
                                "crop_filename": crop_filename,
                                "status": "skipped_no_vlm",
                                "duration_sec": 0.0,
                            },
                            "prep_sec": 0.0,
                            "infer_sec": 0.0,
                            "vlm_inc": 0,
                            "failed_inc": 0,
                        }

                    vlm_inc = 1
                    if not crop_path or not crop_path.exists():
                        return {
                            "result_block": {
                                "label": label,
                                "order": block.get("order", 0),
                                "bbox": block.get("bbox"),
                                "crop_filename": crop_filename,
                                "status": "missing_crop",
                                "content": "",
                                "raw_content": "",
                            },
                            "crop_row": {
                                "image_id": image_id,
                                "label": label,
                                "crop_filename": crop_filename,
                                "status": "missing_crop",
                                "duration_sec": 0.0,
                            },
                            "prep_sec": 0.0,
                            "infer_sec": 0.0,
                            "vlm_inc": vlm_inc,
                            "failed_inc": 1,
                        }

                    prep_t0 = time.time()
                    crop_image = Image.open(crop_path).convert("RGB")
                    model_input_image = _resize_image(crop_image, elem_cfg["resize_min_side"])
                    prep_sec = time.time() - prep_t0

                    infer_status = "ok"
                    raw_text = ""
                    output_tokens = 0
                    infer_sec = 0.0
                    failed_inc = 0
                    error_text = ""
                    try:
                        infer_t0 = time.time()
                        pred = engine.predict_crop(
                            model_input_image,
                            prompt=elem_cfg["prompt"],
                            max_new_tokens=elem_cfg["max_new_tokens"],
                            slice_nums=elem_cfg["slice_nums"],
                        )
                        infer_sec = time.time() - infer_t0
                        raw_text = pred["text"]
                        output_tokens = pred.get("output_token_count", 0)
                    except Exception as e:
                        infer_status = "vllm_error"
                        raw_text = ""
                        failed_inc = 1
                        error_text = str(e)
                    finally:
                        if model_input_image is not crop_image:
                            model_input_image.close()
                        crop_image.close()

                    content = postprocess_content(raw_text, elem_cfg["output_format"], label=label)
                    crop_row = {
                        "image_id": image_id,
                        "label": label,
                        "crop_filename": crop_filename,
                        "status": infer_status,
                        "duration_sec": round(time.time() - crop_started, 6),
                    }
                    if infer_status != "ok":
                        crop_row["error"] = error_text

                    return {
                        "result_block": {
                            "label": label,
                            "order": block.get("order", 0),
                            "bbox": block.get("bbox"),
                            "crop_filename": crop_filename,
                            "status": infer_status,
                            "content": content,
                            "raw_content": raw_text,
                            "output_token_count": output_tokens,
                        },
                        "crop_row": crop_row,
                        "prep_sec": prep_sec,
                        "infer_sec": infer_sec,
                        "vlm_inc": vlm_inc,
                        "failed_inc": failed_inc,
                    }

                if block_workers == 1 or len(blocks) <= 1:
                    block_outputs = [_run_block(block) for block in blocks]
                else:
                    block_outputs_by_idx = {}
                    with ThreadPoolExecutor(max_workers=min(block_workers, len(blocks))) as executor:
                        futures = {executor.submit(_run_block, block): i for i, block in enumerate(blocks)}
                        for future in as_completed(futures):
                            block_outputs_by_idx[futures[future]] = future.result()
                    block_outputs = [block_outputs_by_idx[i] for i in range(len(blocks))]

                for item in block_outputs:
                    result_blocks.append(item["result_block"])
                    extract_prep_sec += float(item["prep_sec"])
                    extract_infer_sec += float(item["infer_sec"])
                    num_vlm_blocks += int(item["vlm_inc"])
                    num_failed_blocks += int(item["failed_inc"])
                    crop_rows.append(item["crop_row"])

                post_t0 = time.time()
                final_blocks = _normalize_and_dedupe_blocks(result_blocks)
                markdown = assemble_markdown(final_blocks, prompt_map)
                post_process_sec += time.time() - post_t0

                ensure_dir(md_path.parent)
                ensure_dir(json_path.parent)
                md_path.write_text(markdown, encoding="utf-8")
                result_obj = {
                    "image_id": image_id,
                    "image_width": layout.get("image_width"),
                    "image_height": layout.get("image_height"),
                    "source": layout.get("source"),
                    "crop_blocks": final_blocks,
                    "markdown": markdown,
                    "result_stage": "infer_final",
                }
                write_json(json_path, result_obj)

                status = "ok" if num_failed_blocks == 0 else "partial_failed"
                row = {
                    "image_id": image_id,
                    "status": status,
                    "page_total_wall_sec": round(time.time() - page_started, 6),
                    "layout_stage_sec": round(layout_stage_sec, 6),
                    "extract_prep_sec": round(extract_prep_sec, 6),
                    "extract_infer_sec": round(extract_infer_sec, 6),
                    "post_process_sec": round(post_process_sec, 6),
                    "num_blocks": len(blocks),
                    "num_vlm_blocks": num_vlm_blocks,
                    "num_failed_blocks": num_failed_blocks,
                    "output_chars": len(markdown),
                    "error": "",
                }
                page_timing = {
                    "image_id": image_id,
                    "status": status,
                    "duration_sec": row["page_total_wall_sec"],
                    "num_blocks": len(blocks),
                    "num_vlm_blocks": num_vlm_blocks,
                    "num_failed_blocks": num_failed_blocks,
                }
                return {"page_row": row, "page_timing": page_timing, "crop_rows": crop_rows}
            except Exception as e:
                row = {
                    "image_id": image_id,
                    "status": "failed",
                    "page_total_wall_sec": round(time.time() - page_started, 6),
                    "layout_stage_sec": round(layout_stage_sec, 6),
                    "extract_prep_sec": round(extract_prep_sec, 6),
                    "extract_infer_sec": round(extract_infer_sec, 6),
                    "post_process_sec": round(post_process_sec, 6),
                    "num_blocks": 0,
                    "num_vlm_blocks": num_vlm_blocks,
                    "num_failed_blocks": num_failed_blocks + 1,
                    "output_chars": 0,
                    "error": str(e),
                }
                page_timing = {
                    "image_id": image_id,
                    "status": "failed",
                    "duration_sec": row["page_total_wall_sec"],
                    "error": str(e),
                }
                return {"page_row": row, "page_timing": page_timing, "crop_rows": crop_rows}

        page_results_by_idx = {}
        if page_workers == 1 or len(tasks) <= 1:
            for idx, task in enumerate(tasks, start=1):
                page_results_by_idx[idx - 1] = _process_page(task)
                if idx % progress_every == 0:
                    print(f"[infer] progress {idx}/{len(tasks)}")
        else:
            completed = 0
            with ThreadPoolExecutor(max_workers=min(page_workers, len(tasks))) as executor:
                futures = {executor.submit(_process_page, task): i for i, task in enumerate(tasks)}
                for future in as_completed(futures):
                    task_idx = futures[future]
                    try:
                        page_results_by_idx[task_idx] = future.result()
                    except Exception as e:
                        image_id = tasks[task_idx].get("image_id", f"task_{task_idx}")
                        failed_row = {
                            "image_id": image_id,
                            "status": "failed",
                            "page_total_wall_sec": 0.0,
                            "layout_stage_sec": 0.0,
                            "extract_prep_sec": 0.0,
                            "extract_infer_sec": 0.0,
                            "post_process_sec": 0.0,
                            "num_blocks": 0,
                            "num_vlm_blocks": 0,
                            "num_failed_blocks": 1,
                            "output_chars": 0,
                            "error": str(e),
                        }
                        page_results_by_idx[task_idx] = {
                            "page_row": failed_row,
                            "page_timing": {"image_id": image_id, "status": "failed", "duration_sec": 0.0, "error": str(e)},
                            "crop_rows": [],
                        }
                    completed += 1
                    if completed % progress_every == 0 or completed == len(tasks):
                        print(f"[infer] progress {completed}/{len(tasks)}")

        for idx in range(len(tasks)):
            result = page_results_by_idx[idx]
            per_page.append(result["page_row"])
            for crop_row in result.get("crop_rows", []):
                append_jsonl(crop_timing_path, crop_row)
            append_jsonl(page_timing_path, result["page_timing"])
    finally:
        engine.close()

    metric_definition = read_json(run_meta_path, default={}).get("metric_definition", {})
    write_per_page_metrics(run_dir, per_page)
    summary = build_summary(per_page, planned_count=len(tasks), metric_definition=metric_definition)
    write_summary(run_dir, summary)

    _update_run_meta(
        run_meta_path,
        {
            "infer_finished_at": now_iso(),
            "counts": {
                "planned_pages": len(tasks),
                "ok_pages": sum(1 for x in per_page if x["status"] == "ok"),
                "failed_pages": sum(1 for x in per_page if x["status"] != "ok"),
            },
            "infer_summary": summary,
        },
    )
    return summary
