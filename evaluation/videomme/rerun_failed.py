"""
扫描 output JSON 中 response 不合法的题目并重跑推理。
用法：python rerun_failed.py [--gpu 0] [--output-json path]
"""
import json
import logging
import argparse

import pandas as pd

from eval_cpp_config import (
    PARQUET_PATH, VIDEO_DATA_DIR, USER_PROMPT_TEMPLATE,
)
from eval_cpp_cli_client import OmniCliClient
from eval_cpp_video_prep import prepare_video_frames, cleanup_frames
from eval_cpp_pipeline import extract_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_ANSWERS = {"A", "B", "C", "D"}


def find_failed_qids(output_json: str) -> set:
    """扫描 output JSON，找出 response 不是 A/B/C/D 的 question_id。"""
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    failed = set()
    for v in data:
        for q in v["questions"]:
            if q.get("response", "") not in VALID_ANSWERS:
                failed.add(q["question_id"])
                logger.info(f"  Invalid: qid={q['question_id']}  response='{q.get('response', '')}'")
    return failed


def load_questions_by_qids(qids: set):
    """从 parquet 中加载指定 question_id 的题目信息。"""
    df = pd.read_parquet(PARQUET_PATH)
    questions = []
    for _, row in df.iterrows():
        if row["question_id"] in qids:
            opts = row["options"]
            if hasattr(opts, "tolist"):
                opts = opts.tolist()
            questions.append({
                "question_id": row["question_id"],
                "video_id": row["video_id"],
                "videoID": row["videoID"],
                "task_type": row["task_type"],
                "question": row["question"],
                "options": opts,
                "answer": row["answer"],
            })
    return questions


def rerun_questions(client, questions):
    """重跑指定题目，返回 {question_id: pred_letter}。"""
    results = {}
    for q in questions:
        qid = q["question_id"]
        vid = q["video_id"]
        video_path = f"{VIDEO_DATA_DIR}/{q['videoID']}.mp4"

        logger.info(f"--- {qid} (video {vid}) ---")
        frame_paths = prepare_video_frames(video_path, vid)
        if not frame_paths:
            logger.error(f"No frames for {vid}")
            results[qid] = ""
            continue

        try:
            options_text = "\n".join(q["options"])
            prompt = USER_PROMPT_TEMPLATE.format(question=q["question"], options=options_text)
            raw = client.infer(frame_paths, prompt, qid=qid)
        except Exception as e:
            logger.error(f"Error: {e}")
            raw = f"[ERROR] {e}"

        pred = extract_answer(raw)
        logger.info(f"  GT={q['answer']}  Raw='{raw}'  Pred='{pred}'")
        results[qid] = pred
        cleanup_frames(vid)

    return results


def patch_output(output_json: str, results: dict):
    """将重跑结果回写到 output JSON。"""
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    patched = 0
    for v in data:
        for qq in v["questions"]:
            if qq["question_id"] in results:
                old = qq["response"]
                qq["response"] = results[qq["question_id"]]
                patched += 1
                logger.info(f"  Patched {qq['question_id']}: '{old}' -> '{qq['response']}'")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"Patched {patched} questions in {output_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-json", default="output/output_videomme_cpp.json")
    args = parser.parse_args()

    # 1. 扫描不合法的 response
    logger.info(f"Scanning {args.output_json} for invalid responses...")
    failed_qids = find_failed_qids(args.output_json)
    if not failed_qids:
        logger.info("All responses are valid, nothing to rerun.")
        return
    logger.info(f"Found {len(failed_qids)} invalid responses: {sorted(failed_qids)}")

    # 2. 从 parquet 加载对应题目
    questions = load_questions_by_qids(failed_qids)
    logger.info(f"Loaded {len(questions)} questions from parquet")

    # 3. 启动 CLI 进程
    logger.info(f"Starting CLI on GPU {args.gpu}...")
    client = OmniCliClient(args.gpu)
    if not client.wait_ready():
        client.close()
        raise RuntimeError("CLI failed to start")

    try:
        # 4. 重跑
        results = rerun_questions(client, questions)
    finally:
        client.close()

    # 5. 回写结果
    patch_output(args.output_json, results)

    # 6. 再检查一遍
    still_failed = {qid for qid, pred in results.items() if pred not in VALID_ANSWERS}
    if still_failed:
        logger.warning(f"Still invalid after rerun: {sorted(still_failed)}")
    else:
        logger.info("All rerun responses are now valid.")


if __name__ == "__main__":
    main()
