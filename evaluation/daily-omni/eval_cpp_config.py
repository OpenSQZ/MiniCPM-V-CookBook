"""
Daily-Omni CPP 评测 Pipeline 配置（CLI 版）
"""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ==================== 路径配置 ====================

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# llama-omni-eval-daily-cli 可执行文件（pipe 驱动的批量推理 CLI，替代旧的 HTTP llama-server）
LLAMA_CLI_BIN = os.environ.get(
    "LLAMA_CLI_BIN",
    os.path.join(PROJ_ROOT, "llama.cpp-omni", "build-eval", "bin", "llama-omni-eval-daily-cli"),
)

# 模型文件（vision/audio/tts 会从 LLM 所在目录自动推断）
LLM_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    os.path.join(PROJ_ROOT, "llama.cpp-omni", "tools", "omni", "convert", "gguf", "llm", "MiniCPM-o-4_5-llm-Q4_K_M.gguf"),
)
GGUF_MODEL_DIR = os.environ.get(
    "GGUF_MODEL_DIR",
    os.path.join(PROJ_ROOT, "llama.cpp-omni", "tools", "omni", "convert", "gguf"),
)

# 数据集
DATASET_DIR = os.environ.get(
    "DATASET_DIR",
    os.path.join(os.path.expanduser("~"), "daily-omni"),
)
ANNOTATION_PATH = os.environ.get(
    "ANNOTATION_PATH",
    os.path.join(DATASET_DIR, "daily_omni.jsonl"),
)

# 输出
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "output_daily_omni_cpp.json")

# 临时文件目录（帧 JPG + 音频 WAV 片段）
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_media")
FRAME_TEMP_DIR = os.path.join(TEMP_DIR, "frames")
AUDIO_TEMP_DIR = os.path.join(TEMP_DIR, "audio_segments")

# ==================== CLI 进程配置 ====================

NUM_GPUS = int(os.environ.get("NUM_GPUS", "8"))
CTX_SIZE = int(os.environ.get("CTX_SIZE", "40960"))

# ==================== 评测参数（对齐 evalkit Daily-Omni 配置）====================

MAX_NUM_FRAMES = 64
MAX_FPS = 1.0
MAX_SLICE_NUMS = 0          # 视频场景不分块，cpp参数0代表不分块，跟python推理有区别
MAX_TOKENS = 128
AUDIO_SR = 16000             # 音频采样率

TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 100
REPEAT_PENALTY = 1.02

# 注：CLI 侧固定 media_type=2（audio+vision）、use_tts=false，无需从这里配置。

# ==================== Prompt 模板（对齐 omni_generation_configs_nosys_interleave.json）====================

USER_PROMPT_TEMPLATE = (
    "Carefully read the following question and select the letter corresponding to the correct answer."
    "Highlight the applicable choices without giving explanations.\n"
    "{question}\n"
    "Options:\n{options}\n"
    "Please select the correct answer from the options above. Only respond with the letter."
)

# ==================== 超时与重试 ====================

CLI_STARTUP_TIMEOUT = 300      # 等待 CLI 加载模型（秒）
INFER_TIMEOUT = 300            # 单题推理超时（秒），多帧+多音频 prefill + decode 可能较慢
