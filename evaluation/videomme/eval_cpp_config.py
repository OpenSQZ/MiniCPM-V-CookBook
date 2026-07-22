"""
Video-MME CPP 评测 Pipeline 配置
"""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ==================== 路径配置 ====================

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# llama-omni-eval-cli 可执行文件（pipe 驱动的批量推理 CLI，替代旧的 HTTP llama-server）
LLAMA_CLI_BIN = os.environ.get(
    "LLAMA_CLI_BIN",
    os.path.join(PROJ_ROOT, "llama.cpp-omni", "build-eval", "bin", "llama-omni-eval-cli"),
)

# 模型文件
LLM_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    os.path.join(PROJ_ROOT, "llama.cpp-omni", "tools", "omni", "convert", "gguf", "llm", "MiniCPM-o-4_5-llm-Q4_K_M.gguf"),
)
GGUF_MODEL_DIR = os.environ.get(
    "GGUF_MODEL_DIR",
    os.path.join(PROJ_ROOT, "llama.cpp-omni", "tools", "omni", "convert", "gguf"),
)

# 数据集
PARQUET_PATH = os.environ.get(
    "PARQUET_PATH",
    os.path.join(PROJ_ROOT, "Video-MME", "videomme", "test-00000-of-00001.parquet"),
)
VIDEO_DATA_DIR = os.environ.get(
    "VIDEO_DATA_DIR",
    os.path.join(PROJ_ROOT, "Video-MME", "data"),
)

# 输出
OUTPUT_DIR = os.path.join(PROJ_ROOT, "videomme", "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "output_videomme_cpp.json")

# 临时帧文件目录
FRAME_TEMP_DIR = os.path.join(PROJ_ROOT, "videomme", "tmp_frames")

# ==================== CLI 进程配置 ====================

NUM_GPUS = int(os.environ.get("NUM_GPUS", "8"))
CTX_SIZE = int(os.environ.get("CTX_SIZE", "40960"))

# ==================== 评测参数 ====================

MAX_NUM_FRAMES = 64
MAX_FPS = 1.0
MAX_SLICE_NUMS = 0
MAX_TOKENS = 100          # 每题最多生成 token 数（CLI --n-predict）

# 解码策略：当前 TEMPERATURE=0.0 为 greedy（对齐 Python do_sample=False）
# 若需开启 sampling，将 TEMPERATURE 改为 > 0（如 0.2）
TEMPERATURE = 0.2
TOP_P = 0.8
TOP_K = 100
REPEAT_PENALTY = 1.02

# 注：CLI 侧固定 media_type=2（audio+vision）、use_tts=false，无需从这里配置。

# ==================== Prompt 模板 ====================

USER_PROMPT_TEMPLATE = (
    "Carefully read the following question and select the letter corresponding to the correct answer."
    "Highlight the applicable choices without giving explanations.\n"
    "{question}\n"
    "Options:\n{options}"
)

# ==================== 超时与重试 ====================

CLI_STARTUP_TIMEOUT = 300      # 等待 CLI 加载模型（秒）
INFER_TIMEOUT = 300            # 单题推理超时（秒），多帧 prefill + decode 可能较慢
