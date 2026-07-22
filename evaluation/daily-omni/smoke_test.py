"""
最小 smoke test（Daily-Omni CLI 版）：从 JSONL 取前 N 条，采样对应视频帧 + 音频段，
用 OmniCliClient 交错 prefill 跑一遍。不评分、不重跑、只验证 CLI + 管道协议端到端可用。

用法：python smoke_test.py [N]   （默认 2）
"""
import sys
import os
import logging

from eval_cpp_config import DATASET_DIR, ANNOTATION_PATH
from eval_cpp_cli_client import OmniCliClient
from eval_cpp_video_prep import prepare_video_frames, cleanup_sample_media
from eval_cpp_audio_prep import prepare_audio_segments
from eval_cpp_pipeline import load_dataset, build_paths, build_prompt, extract_answer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("smoke")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2

samples = load_dataset(ANNOTATION_PATH, limit=N)
log.info(f"Loaded {len(samples)} samples")

client = OmniCliClient(gpu_id=0)
if not client.wait_ready():
    log.error("CLI failed to become ready")
    sys.exit(1)
log.info("CLI ready, running inferences...")

try:
    ok = 0
    for sample in samples:
        sid = sample.get("video_id", "?")
        paths = build_paths(sample, DATASET_DIR)
        video_path, audio_path = paths["video_path"], paths["audio_path"]

        if not os.path.isfile(video_path):
            log.error(f"[{sid}] video not found: {video_path}")
            continue

        frame_paths, timestamps = prepare_video_frames(video_path, sid)
        if not frame_paths:
            log.error(f"[{sid}] no frames")
            continue

        audio_seg_paths = []
        if os.path.isfile(audio_path):
            audio_seg_paths = prepare_audio_segments(audio_path, timestamps, sid)

        log.info(f"[{sid}] {len(frame_paths)} frames, {len(audio_seg_paths)} audio segs")

        prompt = build_prompt(sample["question"], sample["choices"])
        raw = client.infer(frame_paths, audio_seg_paths, prompt, qid=sid)
        pred = extract_answer(raw)
        log.info(f"  GT={sample['gt_answer']}  Pred={pred!r}  Raw={raw!r}")
        cleanup_sample_media(sid)
        if pred in {"A", "B", "C", "D"}:
            ok += 1
    log.info(f"Smoke test done: {ok}/{len(samples)} produced a valid A/B/C/D answer")
finally:
    client.close()
