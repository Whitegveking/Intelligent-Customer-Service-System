import os

# 应用配置
APP_NAME = "智能客服系统"
VERSION = "1.0.0"

# 模型配置
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local_models")
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")

# 确保目录存在
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Whisper 配置
WHISPER_MODEL_SIZE = "small"  # 可选: tiny, base, small, medium, large

# 情感分析配置
EMOTION_MODEL_PATH = "uer/chinese_roberta_L-12_H-768_A-12_E-1-sentiment"
EMOTION_TOKENIZER_PATH = "uer/chinese_roberta_L-12_H-768"

# Qwen 配置
QWEN_MODEL_PATH = "Qwen/Qwen-1_8B-Chat"