import csv
from pathlib import Path
from typing import Dict, List

from .utils import mean, percentile, write_json


PER_PAGE_FIELDS = [
    "image_id",
    "status",
    "page_total_wall_sec",
    "layout_stage_sec",
    "extract_prep_sec",
    "extract_infer_sec",
    "post_process_sec",
    "num_blocks",
    "num_vlm_blocks",
    "num_failed_blocks",
    "output_chars",
    "error",
]


def write_per_page_metrics(run_dir: Path, rows: List[Dict]) -> None:
    jsonl_path = run_dir / "per_page_metrics.jsonl"
    csv_path = run_dir / "per_page_metrics.csv"

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for row in rows:
            jf.write(f"{__import__('json').dumps(row, ensure_ascii=False)}\n")

    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=PER_PAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PER_PAGE_FIELDS})


def build_summary(rows: List[Dict], planned_count: int, metric_definition: Dict) -> Dict:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    fail_rows = [r for r in rows if r.get("status") != "ok"]
    wall = [float(r.get("page_total_wall_sec", 0.0)) for r in ok_rows]

    summary = {
        "counts": {
            "planned": planned_count,
            "ok": len(ok_rows),
            "failed": len(fail_rows),
        },
        "timing_sec": {
            "mean": round(mean(wall), 6),
            "median": round(percentile(wall, 0.5), 6),
            "p90": round(percentile(wall, 0.9), 6),
        },
        "extract": {
            "ocr_calls": sum(int(r.get("num_vlm_blocks", 0)) for r in rows),
            "chars": sum(int(r.get("output_chars", 0)) for r in rows),
        },
        "metric_definition": metric_definition,
    }
    return summary


def write_summary(run_dir: Path, summary: Dict) -> None:
    write_json(run_dir / "summary.json", summary)
