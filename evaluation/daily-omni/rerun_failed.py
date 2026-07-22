"""
扫描 output JSON 中 prediction 不合法的样本并用常驻 CLI 重跑（Daily-Omni CLI 版）。
用法：python rerun_failed.py [--gpu 0] [--output-json path]
"""
import json
import os
import logging
import argparse
from typing import List, Dict, Any

from eval_cpp_config import DATASET_DIR, ANNOTATION_PATH
from eval_cpp_cli_client import OmniCliClient
from eval_cpp_video_prep import prepare_video_frames, cleanup_sample_media
from eval_cpp_audio_prep import prepare_audio_segments
from eval_cpp_pipeline import build_prompt, extract_answer, build_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_ANSWERS = {"A", "B", "C", "D"}


def find_failed(output_json: str) -> List[int]:
    """扫描 output JSON，找出 prediction 不是 A/B/C/D 的样本索引。"""
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    failed = []
    for i, pred in enumerate(predictions):
        p = pred.get("prediction", "")
        if p not in VALID_ANSWERS:
            vid = pred.get("annotation", {}).get("video_id", "?")
            logger.info(f"  Invalid: idx={i} video_id={vid} prediction='{p}'")
            failed.append(i)
    return failed


def load_sample_by_video_id(
    video_id: str,
    annotation_path: str = ANNOTATION_PATH,
) -> Dict[str, Any]:
    """从 JSONL 中找到指定 video_id 的样本。"""
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample.get("video_id") == video_id:
                return sample
    return {}


def rerun_failed_samples(
    client: OmniCliClient,
    output_json: str,
    failed_indices: List[int],
    data_dir: str = DATASET_DIR,
) -> Dict[int, str]:
    """重跑指定索引的样本，返回 {index: prediction}。"""
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    results = {}

    for idx in failed_indices:
        pred = predictions[idx]
        annotation = pred.get("annotation", {})
        video_id = annotation.get("video_id", "?")

        sample = load_sample_by_video_id(video_id)
        if not sample:
            logger.error(f"Sample not found for video_id={video_id}")
            results[idx] = ""
            continue

        paths = build_paths(sample, data_dir)
        video_path = paths["video_path"]
        audio_path = paths["audio_path"]

        logger.info(f"--- Rerunning idx={idx} video_id={video_id} ---")

        try:
            frame_paths, timestamps = prepare_video_frames(video_path, video_id)
            if not frame_paths:
                logger.error(f"No frames for {video_id}")
                results[idx] = ""
                continue

            audio_seg_paths = []
            if os.path.isfile(audio_path):
                audio_seg_paths = prepare_audio_segments(audio_path, timestamps, video_id)

            prompt = build_prompt(sample["question"], sample["choices"])
            raw = client.infer(frame_paths, audio_seg_paths, prompt, qid=video_id)

        except Exception as e:
            logger.error(f"Error: {e}")
            raw = f"[ERROR] {e}"
        finally:
            cleanup_sample_media(video_id)

        pred_letter = extract_answer(raw)
        logger.info(f"  GT={annotation.get('gt_answer', '?')} Raw='{raw}' Pred='{pred_letter}'")
        results[idx] = pred_letter

    return results


def patch_output(output_json: str, results: Dict[int, str]):
    """将重跑结果回写到 output JSON。"""
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    patched = 0
    for idx, new_pred in results.items():
        if 0 <= idx < len(predictions):
            old = predictions[idx].get("prediction", "")
            predictions[idx]["prediction"] = new_pred
            patched += 1
            vid = predictions[idx].get("annotation", {}).get("video_id", "?")
            logger.info(f"  Patched idx={idx} video_id={vid}: '{old}' -> '{new_pred}'")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"Patched {patched} predictions in {output_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-json", default="output/output_daily_omni_cpp.json")
    parser.add_argument("--data-dir", default=DATASET_DIR)
    args = parser.parse_args()

    logger.info(f"Scanning {args.output_json} for invalid predictions...")
    failed_indices = find_failed(args.output_json)
    if not failed_indices:
        logger.info("All predictions are valid, nothing to rerun.")
        return
    logger.info(f"Found {len(failed_indices)} invalid predictions")

    logger.info(f"Starting daily CLI on GPU {args.gpu}...")
    client = OmniCliClient(args.gpu)
    if not client.wait_ready():
        client.close()
        raise RuntimeError("CLI failed to start")

    try:
        results = rerun_failed_samples(client, args.output_json, failed_indices, data_dir=args.data_dir)
    finally:
        client.close()

    patch_output(args.output_json, results)

    still_failed = sum(1 for pred in results.values() if pred not in VALID_ANSWERS)
    if still_failed:
        logger.warning(f"Still invalid after rerun: {still_failed} samples")
    else:
        logger.info("All rerun predictions are now valid.")


if __name__ == "__main__":
    main()
