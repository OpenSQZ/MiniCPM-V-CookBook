"""
最小 smoke test：从 parquet 取前 2 题，采样对应视频帧，用 OmniCliClient 跑一遍。
不评分、不重跑、只验证 CLI + 管道协议端到端可用。
"""
import sys
import logging
import pandas as pd

from eval_cpp_config import PARQUET_PATH, VIDEO_DATA_DIR, USER_PROMPT_TEMPLATE
from eval_cpp_cli_client import OmniCliClient
from eval_cpp_video_prep import prepare_video_frames, cleanup_frames
from eval_cpp_pipeline import extract_answer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("smoke")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2

df = pd.read_parquet(PARQUET_PATH).head(N)
log.info(f"Loaded {len(df)} questions")

client = OmniCliClient(gpu_id=0)
if not client.wait_ready():
    log.error("CLI failed to become ready")
    sys.exit(1)
log.info("CLI ready, running inferences...")

try:
    ok = 0
    for _, row in df.iterrows():
        vid = row["video_id"]
        video_path = f"{VIDEO_DATA_DIR}/{row['videoID']}.mp4"
        frames = prepare_video_frames(video_path, vid)
        log.info(f"[{row['question_id']}] {len(frames)} frames from {video_path}")
        if not frames:
            log.error(f"  no frames for {vid}")
            continue
        opts = row["options"]
        opts = opts.tolist() if hasattr(opts, "tolist") else list(opts)
        prompt = USER_PROMPT_TEMPLATE.format(question=row["question"], options="\n".join(opts))
        raw = client.infer(frames, prompt, qid=row["question_id"])
        pred = extract_answer(raw)
        log.info(f"  GT={row['answer']}  Pred={pred!r}  Raw={raw!r}")
        cleanup_frames(vid)
        if pred in {"A", "B", "C", "D"}:
            ok += 1
    log.info(f"Smoke test done: {ok}/{len(df)} produced a valid A/B/C/D answer")
finally:
    client.close()
