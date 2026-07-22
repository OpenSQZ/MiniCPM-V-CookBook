#!/bin/bash
set -e

# ============================================================
# CPP 版 MiniCPM-o 4.5 TTS Evaluation Script (Chinese / seed-zh)
# ============================================================

# 集中配置：所有外部路径/设备都在 pipeline.env（可用环境变量覆盖）
source "$(cd "$(dirname "$0")"; pwd)/pipeline.env"

echo "Python: $(command -v python3)"

# llama-omni-tts-eval 运行时需要其构建目录下的 libomni.so 等；CUDA 版还需 CUDA 运行库
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}:${CPP_BUILD_LIB}:${LD_LIBRARY_PATH}"

# 1. 参数设置
TIME_STR=$(date +%Y%m%d_%H%M%S)
SEED=${SEED:-42}

# C++ 模型目录（GGUF 格式）
MODEL_PATH=${MODEL_PATH:-"/path/to/MiniCPM-o-4_5-gguf"}
# TTS 模型路径覆盖（可选，用于测试量化版本，留空则由 C++ 从 MODEL_PATH 自动查找）
TTS_MODEL_PATH=${TTS_MODEL_PATH:-"$MODEL_PATH/tts/MiniCPM-o-4_5-tts-F16.gguf"}
# C++ 推理程序路径
CPP_BIN=${CPP_BIN:-"/path/to/llama.cpp-omni/build/bin/llama-omni-tts-eval"}
# ONNX 模型目录（prompt_bundle 提取用）
ONNX_MODEL_DIR=${ONNX_MODEL_DIR:-"/path/to/Step-Audio-2-mini/token2wav"}

SAVE_DIR=${SAVE_DIR:-"${WORKDIR:-$(cd "$(dirname "$0")"; pwd)}/eval_results/cpp-zh-${TIME_STR}-${SEED}"}

LANG="zh"
EVAL_META_PATH=${EVAL_META_PATH:-"/path/to/seedtts_testset_zh/zh/meta.lst"}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-"/path/to/seedtts_testset_zh/zh"}

# WavLM speaker verification checkpoint
SPEAKER_CKPT=${SPEAKER_CKPT:-"/path/to/wavlm_large_finetune.pth"}

echo "============================================"
echo "CPP TTS Evaluation Pipeline"
echo "============================================"
echo "MODEL_PATH:      ${MODEL_PATH}"
echo "TTS_MODEL_PATH:  ${TTS_MODEL_PATH:-(auto from MODEL_PATH)}"
echo "CPP_BIN:         ${CPP_BIN}"
echo "ONNX_MODEL_DIR:  ${ONNX_MODEL_DIR}"
echo "SAVE_DIR:        ${SAVE_DIR}"
echo "EVAL_META_PATH:  ${EVAL_META_PATH}"
echo "EVAL_DATA_PATH:  ${EVAL_DATA_PATH}"
echo "LANGUAGE:        ${LANG}"
echo "SEED:            ${SEED}"
echo "============================================"

mkdir -p "$SAVE_DIR"

WORKDIR=$(cd "$(dirname "$0")"; pwd)
# 无显卡默认单进程（CPU）；有多卡可 GPUS_PER_NODE=8
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
# 只跑前 N 条（冒烟/调试用）；默认全量
NUM_SAMPLES=${NUM_SAMPLES:-10000000}
EVAL_SCRIPT_DIR=${EVAL_SCRIPT_DIR:-"${WORKDIR}/eval_tools"}
SPEAKER_VERIF_DIR=${SPEAKER_VERIF_DIR:-"${EVAL_SCRIPT_DIR}/speaker_verification"}
export PARAFORMER_MODEL
export S3PRL_REPO
export WER_DEVICE

echo "EVAL_SCRIPT_DIR: ${EVAL_SCRIPT_DIR}"
echo "S3PRL_REPO:      ${S3PRL_REPO}"
echo "PARAFORMER:      ${PARAFORMER_MODEL}"

# 统一日志目录：所有 cpp / python 日志都放到 logs/ 下，文件名带时间戳
LOG_BASE="${WORKDIR}/logs"
mkdir -p "$LOG_BASE"

# 构建 TTS_MODEL_PATH 相关的 generate_cpp.py 参数
TTS_ARG=""
if [ -n "$TTS_MODEL_PATH" ]; then
    TTS_ARG="--tts-model-path ${TTS_MODEL_PATH}"
fi

# ============================================================
# 2. prompt_bundle 预提取（只需执行一次，所有 rank 共享）
#    - 单独一步先把所有参考音频的 prompt_bundle 提取好
#    - 后面各 rank 的 generate_cpp.py 会自动跳过已提取的
# ============================================================
echo "=== Step 1: Pre-extract prompt_bundles ==="
BUNDLE_DIR="${SAVE_DIR}/_prompt_bundles"
mkdir -p "$BUNDLE_DIR"

EXTRACT_LOG="${LOG_BASE}/extract_bundle_${TIME_STR}.log"

# 生成去重的 batch list（NUM_SAMPLES 限制时只预提取会用到的前若干条，避免 CPU 上全量提取 1010 条）
python3 -c "
import os, hashlib
meta_path = '${EVAL_META_PATH}'
data_path = '${EVAL_DATA_PATH}'
bundle_dir = '${BUNDLE_DIR}'
num_samples = ${NUM_SAMPLES}
world = ${GPUS_PER_NODE}
cap = num_samples * world if num_samples < 10000000 else None
seen = set()
tasks = []
with open(meta_path) as f:
    for i, line in enumerate(f):
        if cap is not None and i >= cap:
            break
        parts = line.strip().split('|')
        wav_rel = None
        if len(parts) == 5:
            wav_rel = parts[2]
        elif len(parts) == 4:
            wav_rel = parts[2]
        elif len(parts) == 3:
            wav_rel = parts[2]
        if wav_rel and wav_rel not in seen:
            seen.add(wav_rel)
            wav_full = os.path.join(data_path, wav_rel)
            h = hashlib.md5(wav_rel.encode()).hexdigest()[:12]
            out = os.path.join(bundle_dir, h)
            if not os.path.exists(os.path.join(out, 'spk_f32.bin')):
                tasks.append(f'{wav_full}\t{out}')
with open(os.path.join(bundle_dir, '_batch_list.tsv'), 'w') as f:
    f.write('\n'.join(tasks) + '\n')
print(f'Total unique wavs: {len(seen)}, to extract: {len(tasks)}')
"

BATCH_LIST="${BUNDLE_DIR}/_batch_list.tsv"
N_EXTRACT=$(wc -l < "$BATCH_LIST")
if [ "$N_EXTRACT" -gt 0 ]; then
    echo "  Log: ${EXTRACT_LOG}"
    CUDA_VISIBLE_DEVICES=0 python3 "${WORKDIR}/extract_prompt_bundle.py" \
        --batch-list "$BATCH_LIST" \
        --model-dir "$ONNX_MODEL_DIR" \
        --device "${TTS_DEVICE:-cuda}" \
        --skip-existing \
        > "${EXTRACT_LOG}" 2>&1
    echo "=== Prompt bundle extraction done ==="
else
    echo "=== All prompt_bundles already cached ==="
fi

# ============================================================
# 3. TTS 推理 (多 GPU 并行，每个 GPU 启动一个 C++ 进程)
# ============================================================
echo "=== Step 2: C++ TTS Inference (${GPUS_PER_NODE} GPUs) ==="
echo "  Per-GPU logs: ${LOG_BASE}/cpp_gpu*_${TIME_STR}.log"
for i in $(seq 0 $((GPUS_PER_NODE - 1)))
do
    CUDA_VISIBLE_DEVICES=$i python3 "${WORKDIR}/generate_cpp.py" \
        --cpp-bin "${CPP_BIN}" \
        --model-path "${MODEL_PATH}" \
        ${TTS_ARG} \
        --save-dir "${SAVE_DIR}" \
        --eval-meta-path "${EVAL_META_PATH}" \
        --eval-data-path "${EVAL_DATA_PATH}" \
        --onnx-model-dir "${ONNX_MODEL_DIR}" \
        --language "${LANG}" \
        --seed "${SEED}" \
        --temperature 0.3 \
        --teacher-forcing \
        --device "${TTS_DEVICE:-cuda}" \
        --n-gpu-layers "${CPP_NGL:-99}" \
        --t2w-device "${T2W_DEVICE:-gpu:0}" \
        --num-samples "${NUM_SAMPLES}" \
        --rank $i \
        --world-size ${GPUS_PER_NODE} \
        > "${LOG_BASE}/cpp_gpu${i}_${TIME_STR}.log" 2>&1 &
done
wait
echo "=== TTS Inference Done ==="

# ============================================================
# 4. 计算 WER
# ============================================================
echo "=== Step 3: WER Calculation ==="
WER_LOG="${LOG_BASE}/wer_${TIME_STR}.log"
echo "  Log: ${WER_LOG}"
META_LST="$EVAL_META_PATH"
LANGUAGE="$LANG"

WAV_WAV_TEXT=$SAVE_DIR/wav_res_ref_text
SCORE_FILE=$SAVE_DIR/wav_res_ref_text.wer

python3 "${EVAL_SCRIPT_DIR}/get_wav_res_ref_text.py" "$META_LST" "$SAVE_DIR" "$WAV_WAV_TEXT"

timestamp=$(date +%s)
thread_dir=$SAVE_DIR/thread_metas_wer_$timestamp/
mkdir -p "$thread_dir"
num_job=${GPUS_PER_NODE}
num=$(wc -l < "$WAV_WAV_TEXT")
num_per_thread=$(expr $num / $num_job + 1)
split -l $num_per_thread --additional-suffix=.lst -d "$WAV_WAV_TEXT" "$thread_dir/thread-"
out_dir=$thread_dir/results/
mkdir -p "$out_dir"

for rank in $(seq 0 $((num_job - 1))); do
    sub_score_file=$out_dir/thread-0$rank.wer.out
    CUDA_VISIBLE_DEVICES=$rank python3 "${EVAL_SCRIPT_DIR}/run_wer.py" \
        "$thread_dir/thread-0$rank.lst" "$sub_score_file" "$LANGUAGE" \
        >> "${WER_LOG}" 2>&1 &
done
wait

rm -f "$WAV_WAV_TEXT"
rm -f "$SAVE_DIR/merge.out"

cat "$out_dir"/thread-0*.wer.out >> "$out_dir/merge.out"
python3 "${EVAL_SCRIPT_DIR}/average_wer.py" "$out_dir/merge.out" "$SCORE_FILE"

echo "=== WER Calculation Done ==="

# ============================================================
# 5. 计算音频相似度 (Speaker Similarity)
# ============================================================
echo "=== Step 4: Speaker Similarity ==="
SIM_LOG="${LOG_BASE}/sim_${TIME_STR}.log"
echo "  Log: ${SIM_LOG}"
if [ -f "$SPEAKER_CKPT" ]; then
    WAV_WAV_TEXT=$SAVE_DIR/wav_res_ref_text
    SCORE_FILE=$SAVE_DIR/wav_res_ref_text.sim

    python3 "${EVAL_SCRIPT_DIR}/get_wav_res_ref_text.py" "$META_LST" "$SAVE_DIR" "$WAV_WAV_TEXT"

    python3 "${SPEAKER_VERIF_DIR}/verification_pair_list_v2.py" "$WAV_WAV_TEXT" \
        --model_name wavlm_large \
        --checkpoint "$SPEAKER_CKPT" \
        --scores "$SAVE_DIR/wav_res_ref_text.sim.out" \
        --wav1_start_sr 0 \
        --wav2_start_sr 0 \
        --wav1_end_sr -1 \
        --wav2_end_sr -1 \
        --device "${SIM_DEVICE:-cuda:0}" \
        >> "${SIM_LOG}" 2>&1

    rm -f "$WAV_WAV_TEXT"
    rm -f "$SAVE_DIR/merge.out"

    cat "$SAVE_DIR/wav_res_ref_text.sim.out" | grep -v "avg score" >> "$SAVE_DIR/merge.out"
    python3 "${SPEAKER_VERIF_DIR}/average.py" "$SAVE_DIR/merge.out" "$SCORE_FILE"
    echo "=== SIM Calculation Done ==="
else
    echo "WARNING: Speaker checkpoint not found at ${SPEAKER_CKPT}, skipping SIM."
fi

# ============================================================
# 6. 汇总结果
# ============================================================
RESULT_FILE="${WORKDIR}/run_cpp_eval_results.txt"
echo "==============================" >> "$RESULT_FILE"
echo "EVAL DONE: $(date)" >> "$RESULT_FILE"
cat "$SAVE_DIR"/wav_res_ref_text.wer >> "$RESULT_FILE" 2>/dev/null || true
cat "$SAVE_DIR"/wav_res_ref_text.sim >> "$RESULT_FILE" 2>/dev/null || true
echo " MODEL_PATH:      ${MODEL_PATH}" >> "$RESULT_FILE"
echo " TTS_MODEL_PATH:  ${TTS_MODEL_PATH:-(auto)}" >> "$RESULT_FILE"
echo " CPP_BIN:         ${CPP_BIN}" >> "$RESULT_FILE"
echo " SAVE_DIR:        ${SAVE_DIR}" >> "$RESULT_FILE"
echo " EVAL_META_PATH:  ${EVAL_META_PATH}" >> "$RESULT_FILE"
echo " EVAL_DATA_PATH:  ${EVAL_DATA_PATH}" >> "$RESULT_FILE"
echo " SEED:            ${SEED}" >> "$RESULT_FILE"
echo "==============================" >> "$RESULT_FILE"

echo "=== All Evaluation Done ==="
echo "Results saved to: ${SAVE_DIR}"
echo "Logs saved to:    ${LOG_BASE}/*_${TIME_STR}.log"
echo "Summary appended to: ${RESULT_FILE}"
