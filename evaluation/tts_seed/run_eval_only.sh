#!/bin/bash
set -e

# ============================================================
# 仅运行 WER + SIM 评测（跳过推理，直接用已生成的 wav）
# 用法: bash run_eval_only.sh /path/to/save_dir
# ============================================================

source "$(cd "$(dirname "$0")"; pwd)/pipeline.env"

echo "Python: $(command -v python3)"

SAVE_DIR="${1:?用法: bash run_eval_only.sh <save_dir>}"
if [ ! -d "$SAVE_DIR" ]; then
    echo "ERROR: SAVE_DIR not found: $SAVE_DIR"
    exit 1
fi

LANG="zh"
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
EVAL_META_PATH=${EVAL_META_PATH:-"/path/to/seedtts_testset_zh/zh/meta.lst"}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-"/path/to/seedtts_testset_zh/zh"}
SPEAKER_CKPT=${SPEAKER_CKPT:-"/path/to/wavlm_large_finetune.pth"}
WORKDIR=$(cd "$(dirname "$0")"; pwd)
EVAL_SCRIPT_DIR=${EVAL_SCRIPT_DIR:-"${WORKDIR}/eval_tools"}
SPEAKER_VERIF_DIR=${SPEAKER_VERIF_DIR:-"${EVAL_SCRIPT_DIR}/speaker_verification"}
PARAFORMER_MODEL=${PARAFORMER_MODEL:-"/path/to/paraformer-zh"}
S3PRL_REPO=${S3PRL_REPO:-"${EVAL_SCRIPT_DIR}/s3prl-main"}
export PARAFORMER_MODEL
export S3PRL_REPO
export WER_DEVICE

MODEL_PATH=${MODEL_PATH:-"/path/to/MiniCPM-o-4_5-gguf"}
CPP_BIN=${CPP_BIN:-"/path/to/llama.cpp-omni/build/bin/llama-omni-tts-eval"}
SEED=${SEED:-42}
TIME_STR=$(date +%Y%m%d_%H%M%S)

LOG_BASE="${WORKDIR}/logs"
mkdir -p "$LOG_BASE"

echo "============================================"
echo "Eval-only mode"
echo "SAVE_DIR: ${SAVE_DIR}"
echo "EVAL_SCRIPT_DIR: ${EVAL_SCRIPT_DIR}"
echo "S3PRL_REPO: ${S3PRL_REPO}"
echo "PARAFORMER: ${PARAFORMER_MODEL}"
echo "============================================"

wav_count=$(ls "$SAVE_DIR"/*.wav 2>/dev/null | wc -l)
echo "Found ${wav_count} wav files in SAVE_DIR"

# ============================================================
# 1. 计算 WER
# ============================================================
echo "=== Step 1: WER Calculation ==="
WER_LOG="${LOG_BASE}/wer_evalonly_${TIME_STR}.log"
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
# 2. 计算音频相似度 (Speaker Similarity)
# ============================================================
echo "=== Step 2: Speaker Similarity ==="
SIM_LOG="${LOG_BASE}/sim_evalonly_${TIME_STR}.log"
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
# 3. 汇总结果
# ============================================================
RESULT_FILE="${WORKDIR}/run_cpp_eval_results.txt"
echo "==============================" >> "$RESULT_FILE"
echo "EVAL DONE: $(date)" >> "$RESULT_FILE"
cat "$SAVE_DIR"/wav_res_ref_text.wer >> "$RESULT_FILE" 2>/dev/null || true
cat "$SAVE_DIR"/wav_res_ref_text.sim >> "$RESULT_FILE" 2>/dev/null || true
echo " MODEL_PATH:     ${MODEL_PATH}" >> "$RESULT_FILE"
echo " CPP_BIN:        ${CPP_BIN}" >> "$RESULT_FILE"
echo " SAVE_DIR:       ${SAVE_DIR}" >> "$RESULT_FILE"
echo " EVAL_META_PATH: ${EVAL_META_PATH}" >> "$RESULT_FILE"
echo " EVAL_DATA_PATH: ${EVAL_DATA_PATH}" >> "$RESULT_FILE"
echo " SEED:           ${SEED}" >> "$RESULT_FILE"
echo "==============================" >> "$RESULT_FILE"

echo "=== All Evaluation Done ==="
echo "Results saved to: ${SAVE_DIR}"
echo "Logs saved to:    ${LOG_BASE}/*_${TIME_STR}.log"
echo "Summary appended to: ${RESULT_FILE}"
