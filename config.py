"""
アプリケーション設定
"""

import os
from typing import Dict, List


# 利用可能なチャットモデルの定義
AVAILABLE_CHAT_MODELS: Dict[str, Dict] = {
    "qwen2.5-1.5b": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "多言語対応チャットモデル (1.5B) - 日本語対応、チャット専用",
        "language": "multilingual",
        "size": "1.5B",
        "recommended": True,
    },
    "qwen2.5-0.5b": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "description": "超軽量多言語チャットモデル (0.5B) - 日本語対応、高速",
        "language": "multilingual",
        "size": "0.5B",
        "recommended": True,
    },
}

# デフォルトのチャットモデル（環境変数で変更可能）
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5-0.5b")

# チャットサービスの設定
CHAT_CONFIG = {
    "max_history_messages": int(os.getenv("CHAT_MAX_HISTORY", "20")),
    "session_timeout_minutes": int(os.getenv("CHAT_SESSION_TIMEOUT", "60")),
    "max_sessions": int(os.getenv("CHAT_MAX_SESSIONS", "100")),
    "cache_dir": os.getenv("CHAT_MODEL_CACHE_DIR", "./models/chat_llm"),
}


def get_model_info(model_key: str) -> Dict:
    """
    モデルキーから情報を取得

    Args:
        model_key: モデルのキー（例: "tinyllama", "japanese-gpt-neox"）

    Returns:
        モデル情報の辞書
    """
    if model_key not in AVAILABLE_CHAT_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_CHAT_MODELS.keys())}")

    return AVAILABLE_CHAT_MODELS[model_key]


def get_model_name(model_key: str) -> str:
    """
    モデルキーからHugging Face モデル名を取得

    Args:
        model_key: モデルのキー

    Returns:
        Hugging Face モデル名
    """
    return get_model_info(model_key)["name"]


def list_available_models() -> List[Dict]:
    """
    利用可能なモデルのリストを取得

    Returns:
        モデル情報のリスト
    """
    models = []
    for key, info in AVAILABLE_CHAT_MODELS.items():
        model_data = {"key": key, **info, "is_default": key == DEFAULT_CHAT_MODEL}
        models.append(model_data)
    return models
