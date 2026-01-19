"""
Qwen2.5-7B-Instruct モデルをダウンロードして OpenVINO 形式に変換するスクリプト
"""


import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_qwen_7b():
    """Qwen2.5-7B-Instruct をダウンロード"""

    cache_dir = Path("./models/chat_llm")
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    target_dir = cache_dir / "Qwen_Qwen2.5-7B-Instruct"

    # 既存チェック
    if target_dir.exists() and list(target_dir.glob("*.xml")):
        logger.info(f"✓ {model_id} は既にダウンロード済みです")
        return True

    try:
        logger.info(f"📥 {model_id} をダウンロード中...")
        logger.info("⚠️  これには 15-30 分かかる場合があります...")

        # HuggingFace からダウンロード
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        logger.info(f"✓ {model_id} のダウンロードが完了しました")

        # OpenVINO 変換が必要な場合、別途スクリプトを実行
        logger.info("📝 注意: OpenVINO 形式への変換が必要な場合は、別途実行してください")
        logger.info("   参考: https://docs.openvino.ai/latest/notebooks/llm-qwen-convert-with-optimum.html")

        return True

    except Exception as e:
        logger.error(f"❌ ダウンロード失敗: {e}")
        return False


if __name__ == "__main__":
    success = download_qwen_7b()
    sys.exit(0 if success else 1)
