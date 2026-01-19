"""
モデル品質テスト用スクリプト
OpenVINO変換モデル vs 元のHuggingFaceモデルの比較
"""

import sys
import logging

# sys.path操作後のimportはE402を回避するためコメントで説明
sys.path.insert(0, "/home/yokoo/www/openvino-ai-toolkit")

from chat_service import ChatService  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_with_openvino():
    """OpenVINO変換モデルでテスト"""
    logger.info("=" * 60)
    logger.info("OpenVINOモデルでテスト")
    logger.info("=" * 60)

    service = ChatService(model_name="Qwen/Qwen2.5-1.5B-Instruct", use_mock=False)

    test_prompt = """あなたはプロのAIデザイナーです。
次のプロンプトを改善してください。
レスポンスは、日本語でプロンプトのみをテキストで返してください。見出しも不要。
改善したいプロンプト：日本のダンススタジオ、大きな鏡、木製フロア、バー、明るい照明、エネルギッシュ、プロフェッショナル、高品質、8k、広々とした空間
問題：鏡の中がおかしい。変な手・足"""

    result = service.chat(test_prompt, task_type="image_prompt_improvement")
    logger.info(f"応答: {result['response']}")
    return result["response"]


def test_with_mock():
    """モックモードでテスト"""
    logger.info("=" * 60)
    logger.info("モックモードでテスト")
    logger.info("=" * 60)

    service = ChatService(use_mock=True)

    test_prompt = """あなたはプロのAIデザイナーです。
次のプロンプトを改善してください。
レスポンスは、プロンプトのみを返してください。"""

    result = service.chat(test_prompt, task_type="image_prompt_improvement")
    logger.info(f"応答: {result['response']}")
    return result["response"]


if __name__ == "__main__":
    logger.info("\n【テスト開始】\n")

    try:
        openvino_result = test_with_openvino()
        logger.info(f"\nOpenVINO結果:\n{openvino_result}\n")
    except Exception as e:
        logger.error(f"OpenVINOテスト失敗: {e}")

    mock_result = test_with_mock()
    logger.info(f"\nモック結果:\n{mock_result}\n")

    logger.info("テスト完了")
