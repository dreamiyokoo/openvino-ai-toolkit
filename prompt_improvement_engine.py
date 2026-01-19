"""
プロンプト改善専用エンジン
小さいLLMの限界を超える専門的な処理ロジック
"""

import re
import logging

logger = logging.getLogger(__name__)


class PromptImprovementEngine:
    """画像生成プロンプト改善の専門エンジン"""

    @staticmethod
    def extract_original_prompt(user_message: str) -> tuple[str, str]:
        """ユーザーメッセージからプロンプトと問題を抽出"""
        # パターン1: 「改善したいプロンプト：」と「問題：」で区切られている場合
        prompt_match = re.search(r"改善したいプロンプト：(.+?)(?:問題：|$)", user_message, re.DOTALL)
        problem_match = re.search(r"問題：(.+?)$", user_message, re.DOTALL)

        prompt = prompt_match.group(1).strip() if prompt_match else ""
        problem = problem_match.group(1).strip() if problem_match else ""

        return prompt, problem

    @staticmethod
    def improve_prompt(original_prompt: str, problem: str = "") -> str:
        """プロンプトを改善（ルールベース）"""

        # 元のプロンプトを日本語のみに正規化
        prompt = original_prompt.strip()

        # 末尾のコンマを削除
        prompt = prompt.rstrip("、,")

        # 問題点の解決
        if problem:
            # 「鏡の中がおかしい」などの問題を解決する表現を追加
            if "鏡" in problem:
                prompt = re.sub(r"大きな鏡", "大きな鏡（鏡に映る人物の手足が正確で自然な姿勢）", prompt)
                if "大きな鏡" not in prompt:
                    prompt += "、鏡に正確に映る人物の姿勢"

        # フォーマット改善
        parts = [p.strip() for p in prompt.split("、") if p.strip()]

        # プロンプト部品を整理
        improved_parts = []
        added_corrections = set()

        for part in parts:
            improved_parts.append(part)

            # 一般的な改善を自動追加
            if "高品質" in part and "超詳細" not in prompt:
                if "超詳細" not in added_corrections:
                    improved_parts.append("超詳細な描写")
                    added_corrections.add("超詳細")

        # 改善されたプロンプトを生成
        improved = "、".join(improved_parts)

        # 日本語のみを確保
        improved = re.sub(r"\s+", "", improved)  # 空白を削除

        logger.info(f"改善されたプロンプト: {improved}")
        return improved

    @staticmethod
    def generate_prompt(description: str) -> str:
        """ユーザーの説明から画像生成プロンプトを生成（ルールベース）"""
        # シンプルなテンプレートを使用してプロンプトを生成
        prompt = description.strip()

        if not prompt or len(prompt) < 2:
            return "画像生成のプロンプト"

        # 句点や句読点の清掃
        prompt = prompt.lstrip("。、")
        prompt = prompt.strip()

        # 既に十分な長さの場合はそのまま使用
        if len(prompt) > 15:
            # 日本語のみを確保
            prompt = re.sub(r"[^ぁ-ん ァ-ヴー一-龥々〆〤ゝゞ、。・ー，、]", "", prompt)
            prompt = prompt.strip().rstrip("。、 ")

            # よくあるキーワードが入っていなければ追加
            if "高品質" not in prompt:
                if "ダンス" in prompt or "教室" in prompt:
                    prompt = f"{prompt}、高品質、明るい照明"

            return prompt

        # よくあるキーワードを追加
        add_details = []

        if "ダンス" in description or "教室" in description or "スタジオ" in description:
            if "高品質" not in prompt:
                add_details.append("高品質")
            if "明るい" not in prompt and "照明" not in prompt:
                add_details.append("明るい照明")

        if add_details:
            prompt = f"{prompt}、{','.join(add_details)}"

        # 日本語のみを確保
        prompt = re.sub(r"[^ぁ-ん ァ-ヴー一-龥々〆〤ゝゞ、。・ー，、]", "", prompt)
        prompt = re.sub(r"[\s、]+$", "", prompt)  # 末尾の空白と句点を削除
        prompt = re.sub(r",\s*", "、", prompt)  # コンマを日本語句点に統一

        logger.info(f"生成されたプロンプト: {prompt}")
        return prompt


def process_prompt_improvement_request(user_message: str, llm_response: str = "") -> str:
    """
    プロンプト改善リクエストを処理

    Args:
        user_message: ユーザーからのメッセージ
        llm_response: LLMからの応答（オプション、信頼性が低い場合）

    Returns:
        改善されたプロンプト
    """

    # ユーザーメッセージからプロンプトと問題を抽出
    original_prompt, problem = PromptImprovementEngine.extract_original_prompt(user_message)

    if not original_prompt:
        logger.warning("プロンプトが抽出できませんでした")
        return "申し訳ありませんが、プロンプトが確認できませんでした。"

    # LLMレスポンスの検証
    if llm_response:
        cleaned = llm_response.strip()

        # 言語混合検出：複数言語が混在しているかをチェック

        # 英字カウント (単語として)
        english_words = len(re.findall(r"\b[a-zA-Z]+\b", cleaned))

        # 中国語（簡体字）カウント（が、ゃ、など日本語と区別）
        chinese_simplified = len(re.findall(r"[\u4e00-\u9fff]", cleaned))

        # 言語混合判定：英字が3個以上含まれている場合、言語混合の可能性が高い
        if english_words >= 2:
            logger.warning(f"言語混合検出（英字 {english_words}個）: {cleaned[:60]} - ルールベース処理に切り替え")
            improved = PromptImprovementEngine.improve_prompt(original_prompt, problem)
            return improved

        # 中国語が10文字以上混在している場合
        if chinese_simplified >= 10:
            logger.warning(f"中国語混合検出: {chinese_simplified}文字 - ルールベース処理に切り替え")
            improved = PromptImprovementEngine.improve_prompt(original_prompt, problem)
            return improved

        # その他の検証
        if len(cleaned) > 10 and len(cleaned) < 500:
            # 説明的な内容が含まれていないか確認
            bad_markers = ["注意", "備考", "ただし", "ですが", "説明", "理由", "詳細", "ます", "ました", "します"]
            has_explanation = any(marker in cleaned for marker in bad_markers)

            if not has_explanation:
                logger.info(f"LLM応答を採用: {cleaned[:50]}")
                return cleaned

    # LLMレスポンスが不十分な場合はルールベースで改善
    logger.info("ルールベースのプロンプト改善を実行")
    improved = PromptImprovementEngine.improve_prompt(original_prompt, problem)

    return improved


def process_prompt_generation_request(user_message: str, llm_response: str = "") -> str:
    """
    プロンプト生成リクエストを処理

    Args:
        user_message: ユーザーからのメッセージ（画像の説明）
        llm_response: LLMからの応答（オプション、信頼性が低い場合）

    Returns:
        生成されたプロンプト
    """

    if not user_message:
        return "申し訳ありませんが、説明が確認できませんでした。"

    # ユーザーメッセージから実際の説明部分を抽出
    # 「プロンプトを作成」「プロンプトを生成」などの指示部分を除去
    message_clean = user_message

    # よくある指示パターンを削除（より広範に）
    instruction_patterns = [
        r"^画像生成.*?ください。?",  # 行頭から
        r"プロンプト.*?返してください",
        r"プロンプト.*?ください",
        r"日本語.*?返してください",
        r"日本語.*?ください",
        r"１行.*?ください",
        r"1行.*?ください",
    ]

    for pattern in instruction_patterns:
        message_clean = re.sub(pattern, "", message_clean)

    # さらにクリーンアップ
    message_clean = message_clean.strip().lstrip("。、")
    message_clean = re.sub(r"^(\.)*", "", message_clean)
    message_clean = message_clean.strip()

    if not message_clean:
        message_clean = user_message

    # LLMレスポンスの検証
    if llm_response:
        cleaned = llm_response.strip()

        # 言語混合検出
        english_words = len(re.findall(r"\b[a-zA-Z]+\b", cleaned))
        english_chars = len(re.findall(r"[a-zA-Z]", cleaned))
        japanese_chars = len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]", cleaned))

        # 言語混合が2個以上の英単語があった場合はフォールバック
        if english_words >= 2:
            logger.warning(f"言語混合検出（英単語 {english_words}個）: フォールバック")
            generated = PromptImprovementEngine.generate_prompt(message_clean)
            return generated

        # 英字とカタカナの混合パターン検出
        if english_chars >= 3 and japanese_chars > 0:
            logger.warning("言語混合検出（英字混合）: フォールバック")
            generated = PromptImprovementEngine.generate_prompt(message_clean)
            return generated

        # 日本語のみで、適切な長さの場合は採用
        if japanese_chars > 5 and len(cleaned) > 5 and len(cleaned) < 300:
            # 説明的な内容が含まれていないか確認
            bad_markers = ["注意", "備考", "ただし", "ですが", "説明", "理由", "です", "ます", "ました"]
            has_explanation = any(marker in cleaned for marker in bad_markers)

            if not has_explanation:
                logger.info(f"LLM応答を採用: {cleaned[:50]}")
                return cleaned

    # LLMレスポンスが不十分な場合はルールベースで生成
    logger.info("ルールベースのプロンプト生成を実行")
    generated = PromptImprovementEngine.generate_prompt(message_clean)

    return generated


if __name__ == "__main__":
    # テスト
    test_message = """あなたはプロのAIデザイナーです。
次のプロンプトを改善してください。
レスポンスは、日本語でプロンプトのみをテキストで返してください。見出しも不要。
改善したいプロンプト：日本のダンススタジオ、大きな鏡、木製フロア、バー、明るい照明、エネルギッシュ、プロフェッショナル、高品質、8k、広々とした空間
問題：鏡の中がおかしい。変な手・足"""

    result = process_prompt_improvement_request(test_message)
    print(f"改善結果: {result}")
