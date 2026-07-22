"""
Video-MME CPP 评测 Pipeline 主控脚本。

功能：
  1. 加载 parquet 数据集，按 video_id 分组（900 视频 × 3 题）
  2. 均匀分配到 N 个 GPU worker
  3. 每卡启动一个常驻 llama-omni-eval-cli 进程（模型只加载一次）
  4. N 线程并发处理视频评测（通过 JSONL 管道逐题推理）
  5. 收集结果，输出 JSON（对齐 output_test_template.json 格式）

用法：
  python eval_cpp_pipeline.py [--num-gpus 8] [--output output.json]
"""
import os
import sys
import json
import time
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import pandas as pd

from eval_cpp_config import (
    PARQUET_PATH, VIDEO_DATA_DIR, OUTPUT_DIR, OUTPUT_JSON,
    NUM_GPUS, USER_PROMPT_TEMPLATE,
)
from eval_cpp_cli_client import OmniCliClient, start_all_clients, stop_all_clients
from eval_cpp_video_prep import prepare_video_frames, cleanup_frames, cleanup_all_frames

logger = logging.getLogger(__name__)


# ==================== 数据集加载 ====================

def load_dataset(parquet_path: str = PARQUET_PATH, limit: int = 0) -> pd.DataFrame:
    """加载 Video-MME parquet 数据集。limit > 0 时只取前 limit 条。"""
    logger.info(f"Loading dataset from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if limit > 0:
        df = df.head(limit)
        logger.info(f"Limited to first {limit} rows")
    logger.info(f"Loaded {len(df)} questions")
    return df


def group_by_video(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    按 video_id 分组，返回：
    {
        video_id: {
            "video_id": "001",
            "duration": "short",
            "domain": "Knowledge",
            "sub_category": "...",
            "videoID": "xxxx",  # YouTube ID → mp4 文件名
            "questions": [
                {"question_id": "001-1", "task_type": "...", "question": "...",
                 "options": [...], "answer": "A"},
                ...
            ]
        }
    }
    """
    groups = {}
    for _, row in df.iterrows():
        vid = row["video_id"]
        if vid not in groups:
            groups[vid] = {
                "video_id": vid,
                "duration": row["duration"],
                "domain": row["domain"],
                "sub_category": row["sub_category"],
                "videoID": row["videoID"],
                "questions": [],
            }
        opts = row["options"]
        if hasattr(opts, "tolist"):
            opts = opts.tolist()
        groups[vid]["questions"].append({
            "question_id": row["question_id"],
            "task_type": row["task_type"],
            "question": row["question"],
            "options": opts,
            "answer": row["answer"],
        })
    logger.info(f"Grouped into {len(groups)} videos")
    return groups


def split_into_chunks(video_groups: Dict, n: int) -> List[List[Dict]]:
    """将视频分组均匀分成 n 份。"""
    all_videos = list(video_groups.values())
    chunks = [[] for _ in range(n)]
    for i, v in enumerate(all_videos):
        chunks[i % n].append(v)
    for i, c in enumerate(chunks):
        logger.info(f"  Chunk {i}: {len(c)} videos")
    return chunks


# ==================== Prompt 构建 ====================

def build_prompt(question: str, options: list) -> str:
    """
    构建评测文本 prompt（对齐 Python videomme.py 格式）。

    options 来自 parquet，格式为 ["A. xxx", "B. xxx", "C. xxx", "D. xxx"]
    Python 原版格式：每个选项前加 \n，即 Options:\nA. xxx\nB. xxx\nC. xxx\nD. xxx
    """
    options_text = "\n".join(options)
    return USER_PROMPT_TEMPLATE.format(question=question, options=options_text)


# ==================== 答案提取 ====================

import re

def extract_answer(response_text: str) -> str:
    """
    从模型输出中提取 A/B/C/D 答案字母。

    策略（按优先级）：
      1. 整体只有一个字母 A-D（可能带句号）→ 直接返回
      2. 匹配独立出现的大写 A/B/C/D（单词边界，非 "Answer" 等单词的一部分）
      3. 都找不到 → 返回空字符串
    """
    text = response_text.strip()
    if not text:
        return ""

    cleaned = text.rstrip(".").strip()
    if len(cleaned) == 1 and cleaned.upper() in "ABCD":
        return cleaned.upper()

    # 匹配独立的 A/B/C/D：前后都不是字母
    match = re.search(r'(?<![a-zA-Z])([A-D])(?![a-zA-Z])', text)
    if match:
        return match.group(1)
    return ""


# ==================== 单视频处理 ====================

def process_video(
    client: "OmniCliClient",
    video_info: Dict[str, Any],
    stop_event: threading.Event,
    video_data_dir: str = VIDEO_DATA_DIR,
) -> List[Dict[str, Any]]:
    """
    处理单个视频的所有题目。

    流程（每道题）：
      1. reset KV cache
      2. 逐帧 prefill 图片
      3. prefill 文本 prompt
      4. decode 获取模型回答
      5. 提取答案字母
    """
    if stop_event.is_set():
        return []

    video_id = video_info["video_id"]
    video_file = video_info["videoID"] + ".mp4"
    video_path = os.path.join(video_data_dir, video_file)

    if not os.path.isfile(video_path):
        logger.error(f"Video not found: {video_path}")
        return [{
            "question_id": q["question_id"],
            "task_type": q["task_type"],
            "question": q["question"],
            "options": q["options"],
            "answer": q["answer"],
            "response": "[ERROR] video file not found",
        } for q in video_info["questions"]]

    # 帧采样 + 保存临时 JPG（3 道题共享同一组帧）
    frame_paths = prepare_video_frames(video_path, video_id)
    if not frame_paths:
        logger.error(f"No frames extracted from {video_path}")
        return [{
            "question_id": q["question_id"],
            "task_type": q["task_type"],
            "question": q["question"],
            "options": q["options"],
            "answer": q["answer"],
            "response": "[ERROR] no frames extracted",
        } for q in video_info["questions"]]

    results = []
    try:
        for q in video_info["questions"]:
            if stop_event.is_set():
                logger.info(f"Stop requested, skip remaining questions in video {video_id}")
                break
            try:
                # 单题推理：CLI 内部完成 reset -> prefill 帧 -> prefill 文本 -> decode
                prompt = build_prompt(q["question"], q["options"])
                response_text = client.infer(frame_paths, prompt, qid=q["question_id"])

                # 提取答案字母
                answer_pred = extract_answer(response_text)

            except Exception as e:
                logger.error(f"Error processing question {q['question_id']}: {e}")
                response_text = f"[ERROR] {e}"
                answer_pred = ""

            results.append({
                "question_id": q["question_id"],
                "task_type": q["task_type"],
                "question": q["question"],
                "options": q["options"],
                "answer": q["answer"],
                "response": answer_pred,
            })
            resp_short = repr(response_text)[:80]
            logger.info(f"  Q={q['question_id']} GT={q['answer']} Pred={answer_pred} Response={resp_short}")
    finally:
        # 清理临时帧文件
        cleanup_frames(video_id)
    return results


# ==================== Worker 线程 ====================

def process_chunk(
    gpu_id: int,
    client,
    chunk: List[Dict[str, Any]],
    stop_event: threading.Event,
) -> List[Dict[str, Any]]:
    """
    单个 GPU worker：用常驻 CLI 客户端串行处理分配到的所有视频。
    """
    all_results = []
    total = len(chunk)

    for i, video_info in enumerate(chunk):
        if stop_event.is_set():
            logger.info(f"[GPU {gpu_id}] Stop requested, break chunk loop")
            break
        vid = video_info["video_id"]
        logger.info(f"[GPU {gpu_id}] ({i+1}/{total}) Processing video {vid}")
        t0 = time.time()
        results = process_video(client, video_info, stop_event=stop_event)
        elapsed = time.time() - t0
        all_results.extend(results)
        logger.info(f"[GPU {gpu_id}] ({i+1}/{total}) Video {vid} done, {len(results)} questions, {elapsed:.1f}s")

    return all_results


# ==================== 结果输出 ====================

def format_output(
    results: List[Dict[str, Any]],
    video_groups: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    将结果格式化为 output_test_template.json 格式：
    [
        {
            "video_id": "001",
            "duration": "short",
            "domain": "Knowledge",
            "sub_category": "...",
            "questions": [
                {"question_id": "001-1", ..., "response": "A"},
                ...
            ]
        },
        ...
    ]
    """
    result_map = {r["question_id"]: r for r in results}

    output = []
    for vid in sorted(video_groups.keys()):
        info = video_groups[vid]
        entry = {
            "video_id": info["video_id"],
            "duration": info["duration"],
            "domain": info["domain"],
            "sub_category": info["sub_category"],
            "questions": [],
        }
        for q in info["questions"]:
            qid = q["question_id"]
            r = result_map.get(qid, {})
            entry["questions"].append({
                "question_id": qid,
                "task_type": q["task_type"],
                "question": q["question"],
                "options": q["options"],
                "answer": q["answer"],
                "response": r.get("response", ""),
            })
        output.append(entry)
    return output


def save_results(output: List[Dict], path: str = OUTPUT_JSON):
    """保存结果 JSON。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    logger.info(f"Results saved to {path}")


# ==================== Main ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Video-MME CPP Evaluation Pipeline")
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS, help="Number of GPUs to use")
    parser.add_argument("--parquet", type=str, default=PARQUET_PATH, help="Path to parquet file")
    parser.add_argument("--video-dir", type=str, default=VIDEO_DATA_DIR, help="Video data directory")
    parser.add_argument("--output", type=str, default=OUTPUT_JSON, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=0, help="Only load first N rows from parquet (0 = all)")
    parser.add_argument("--skip-rerun", action="store_true", help="Skip rerun of failed questions")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip scoring after evaluation")
    parser.add_argument("--rerun-gpu", type=int, default=0, help="GPU id for rerun CLI process")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    stop_event = threading.Event()
    interrupted = False
    clients = []

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("Video-MME CPP Evaluation Pipeline")
    logger.info(f"  GPUs: {args.num_gpus}")
    logger.info(f"  Parquet: {args.parquet}")
    logger.info(f"  Video dir: {args.video_dir}")
    logger.info(f"  Output: {args.output}")
    logger.info("=" * 60)

    # 1. 加载数据集
    df = load_dataset(args.parquet, limit=args.limit)
    video_groups = group_by_video(df)
    chunks = split_into_chunks(video_groups, args.num_gpus)

    try:
        # 2. 启动常驻 CLI 进程（每卡一个，模型只加载一次）
        logger.info("Starting llama-omni-eval-cli processes...")
        clients = start_all_clients(args.num_gpus)

        # 3. 并发处理
        logger.info("Starting evaluation...")
        t_start = time.time()
        all_results = []

        pool = ThreadPoolExecutor(max_workers=args.num_gpus)
        futures = {}
        try:
            for gpu_id, chunk in enumerate(chunks):
                fut = pool.submit(process_chunk, gpu_id, clients[gpu_id], chunk, stop_event)
                futures[fut] = gpu_id

            for fut in as_completed(futures):
                gpu_id = futures[fut]
                try:
                    results = fut.result()
                    all_results.extend(results)
                    logger.info(f"GPU {gpu_id} completed: {len(results)} questions")
                except Exception as e:
                    logger.error(f"GPU {gpu_id} failed: {e}")
        except KeyboardInterrupt:
            interrupted = True
            stop_event.set()
            logger.warning("KeyboardInterrupt received, stopping workers and CLI processes...")
            for fut in futures:
                fut.cancel()
            # 先停 CLI，尽快让阻塞中的管道读取失败退出
            stop_all_clients(clients)
            clients = []
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        elapsed = time.time() - t_start
        logger.info(f"Evaluation done: {len(all_results)} questions in {elapsed:.1f}s")

        # 4. 格式化并保存结果
        output = format_output(all_results, video_groups)
        save_results(output, args.output)

        # 5. 简单统计
        correct = sum(1 for r in all_results if extract_answer(r.get("response", "")) == r.get("answer", ""))
        total = len(all_results)
        logger.info(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%" if total > 0 else "No results")

    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        logger.warning("Interrupted by user (Ctrl+C).")
    finally:
        # 6. 清理
        logger.info("Stopping CLI processes...")
        stop_all_clients(clients)
        cleanup_all_frames()

    if interrupted:
        logger.warning("Pipeline interrupted, skip rerun and scoring.")
        return

    # 8. 重跑失败题目
    if not args.skip_rerun:
        from rerun_failed import find_failed_qids, load_questions_by_qids, rerun_questions, patch_output
        failed_qids = find_failed_qids(args.output)
        if failed_qids:
            logger.info(f"Rerunning {len(failed_qids)} failed questions...")
            questions = load_questions_by_qids(failed_qids)
            client = OmniCliClient(args.rerun_gpu)
            if client.wait_ready():
                try:
                    results = rerun_questions(client, questions)
                finally:
                    client.close()
                patch_output(args.output, results)
            else:
                client.close()
                logger.error("Rerun CLI failed to start, skipping.")
        else:
            logger.info("All responses valid, no rerun needed.")

    # 9. 评分
    if not args.skip_scoring:
        from eval_your_result import eval_your_results
        logger.info("Running Video-MME scoring...")
        eval_your_results(
            args.output,
            video_types=["short", "medium", "long"],
            return_categories_accuracy=True,
            return_sub_categories_accuracy=True,
            return_task_types_accuracy=True,
        )

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
