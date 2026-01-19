"""
OpenVINOを使ったチャットサービス
軽量LLMを使用してチャット機能を実現
"""

from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import logging
import uuid
import threading

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
from prompt_improvement_engine import process_prompt_improvement_request, process_prompt_generation_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# タスク固有のシステムプロンプトテンプレート
TASK_PROMPTS = {
    "image_prompt_improvement": """You are a professional image generation prompt editor.

CRITICAL INSTRUCTIONS - MUST FOLLOW EXACTLY:
1. The user will give you a prompt to improve
2. Return ONLY the improved prompt text
3. DO NOT provide explanations
4. DO NOT provide reasons
5. DO NOT provide headers
6. Return ONLY the improved prompt
7. Keep it to 1-2 sentences maximum
8. Do not use markdown formatting
9. Do not include bullet points
10. Do not include any text other than the improved prompt

REQUIREMENTS:
- Use clear, concise language
- Add visual details
- Make it specific for image generation
- Address any mentioned issues
- Keep it natural and readable

OUTPUT:
The improved prompt text only. Just the text. 1-2 sentences. Nothing else. No explanations. No additional text.""",
    "image_prompt_generation": """You are a professional image generation prompt writer.

CRITICAL INSTRUCTIONS - MUST FOLLOW EXACTLY:
1. The user will describe what image they want
2. Create a professional image generation prompt
3. Return ONLY the prompt text in Japanese
4. DO NOT provide explanations
5. DO NOT provide reasons
6. Return ONLY the prompt
7. Keep it to 1-2 sentences maximum
8. Do not use markdown formatting
9. Do not include bullet points
10. Do not include any text other than the prompt

REQUIREMENTS:
- Use clear, concise Japanese language
- Include specific visual details
- Make it suitable for image generation AI
- Be descriptive but concise
- Include lighting, composition, style when relevant

OUTPUT:
The image prompt text only. Japanese only. 1-2 sentences. Nothing else.""",
    "general": """You are a helpful and professional assistant.
Answer user questions accurately and clearly.
If you're unsure about something, say so.
Use the user's language for responses.""",
}


class ChatService:
    """OpenVINOベースのチャットサービス（複数モデル対応）"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",  # 7Bモデルが最も高性能
        cache_dir: str = "./models/chat_llm",
        max_history_messages: int = 20,
        session_timeout_minutes: int = 60,
        max_sessions: int = 100,
        use_mock: bool = False,  # モックモードを追加
    ):
        """
        チャットサービスの初期化

        Args:
            model_name: 使用するLLMモデル名
            cache_dir: モデルキャッシュディレクトリ
            max_history_messages: セッションごとの最大メッセージ履歴数
            session_timeout_minutes: セッションタイムアウト（分）
            max_sessions: 最大セッション数
            use_mock: モックモードを使用（開発・デモ用）
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_messages = max_history_messages
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_sessions = max_sessions
        self.use_mock = use_mock

        # モデルとトークナイザー（複数モデルをキャッシュ）
        self.models: Dict[str, any] = {}  # model_name -> model
        self.tokenizers: Dict[str, any] = {}  # model_name -> tokenizer
        self.models_lock = threading.Lock()

        # セッション管理（メモリ内）
        # session_id -> {"messages": [...], "system_prompt": str, "model_name": str, "created_at": datetime, "last_access": datetime}
        self.sessions: Dict[str, Dict] = {}
        self.sessions_lock = threading.Lock()

        # デフォルトモデルをロード（モックモードでない場合のみ）
        if not self.use_mock:
            self._load_model(self.model_name)
        else:
            logger.info("Chat service initialized in MOCK mode")

    def _load_model(self, model_name: str = None):
        """LLMモデルをロード"""
        if model_name is None:
            model_name = self.model_name

        # 既にロード済みの場合はスキップ
        with self.models_lock:
            if model_name in self.models:
                logger.info(f"Model already loaded: {model_name}")
                return

        try:
            logger.info(f"Loading chat model: {model_name}")
            model_path = self.cache_dir / model_name.replace("/", "_")

            # モデルが既にエクスポートされているか確認
            if not model_path.exists():
                logger.info("Exporting chat model to OpenVINO format...")
                model = OVModelForCausalLM.from_pretrained(model_name, export=True, compile=True)
                model.save_pretrained(model_path)
            else:
                logger.info("Loading cached OpenVINO chat model...")
                model = OVModelForCausalLM.from_pretrained(model_path, compile=True)

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # パディングトークンの設定
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            with self.models_lock:
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer

            logger.info(f"Chat model loaded successfully: {model_name}")

        except Exception as e:
            logger.error(f"Error loading chat model {model_name}: {e}")
            raise

    def get_loaded_models(self) -> List[str]:
        """ロード済みのモデル一覧を取得"""
        with self.models_lock:
            return list(self.models.keys())

    def _detect_task_type(self, message: str) -> str:
        """
        メッセージの内容からタスクタイプを検出

        Args:
            message: ユーザーメッセージ

        Returns:
            タスクタイプ（"image_prompt_improvement", "image_prompt_generation", "general" など）
        """
        message_lower = message.lower()

        # 画像プロンプト生成の検出（「プロンプトを作成」「プロンプトを生成」など）
        generation_keywords = ["プロンプト作成", "プロンプト生成", "プロンプトを作", "プロンプトを生", "プロンプト欲しい", "プロンプトください"]
        if any(keyword in message_lower for keyword in generation_keywords):
            return "image_prompt_generation"
        if "create" in message_lower and "prompt" in message_lower:
            return "image_prompt_generation"
        if "generate" in message_lower and "prompt" in message_lower:
            return "image_prompt_generation"

        # 画像プロンプト改善の検出
        if "改善したいプロンプト" in message or "改善" in message and "プロンプト" in message:
            return "image_prompt_improvement"
        if "prompt" in message_lower and "improve" in message_lower:
            return "image_prompt_improvement"

        return "general"

    def _get_system_prompt_for_task(self, task_type: str, custom_prompt: Optional[str] = None, model_name: str = None) -> str:
        """
        タスクタイプに応じたシステムプロンプトを取得

        Args:
            task_type: タスクタイプ
            custom_prompt: カスタムプロンプト（優先）
            model_name: モデル名

        Returns:
            適切なシステムプロンプト
        """
        if custom_prompt:
            return custom_prompt
        return TASK_PROMPTS.get(task_type, TASK_PROMPTS["general"])

    def _post_process_response(self, response: str, task_type: str = "general") -> str:
        """
        生成された応答を後処理してクリーンアップ

        Args:
            response: 生成された応答テキスト
            task_type: タスクタイプ

        Returns:
            クリーンアップされた応答
        """
        # 画像プロンプト生成・改善タスク用の後処理
        if task_type in ["image_prompt_improvement", "image_prompt_generation"]:
            text = response.strip()

            # 説明的な段落を検出・削除
            explanation_markers = [
                "しかし",
                "ただし",
                "ただ",
                "ところで",
                "つまり",
                "注意",
                "注：",
                "注意：",
                "備考",
                "※",
                "⚠",
                "ご了承",
                "可能性",
                "可能",
                "おそらく",
                "と思われ",
                "と考えられ",
                "かもしれません",
                "この改善",
                "改善内容",
                "改善点",
                "理由",
                "映像解像",
                "光源",
                "技術的",
                "撮影",
                "一部で",
            ]

            # 説明的な段落を分割
            sentences = text.split("。")
            result_sentences = []

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # 説明的なマーカーがあればスキップ
                if any(marker in sentence for marker in explanation_markers):
                    continue

                result_sentences.append(sentence)

            # 最初の1-2文のみを使用
            if result_sentences:
                final_text = result_sentences[0]
                if len(result_sentences) > 1 and len(final_text) < 50:
                    final_text += "。" + result_sentences[1]
                elif not final_text.endswith("。"):
                    final_text += "。"

                # 英語とローマ字の混在を日本語に統一
                replacements = {
                    "STUDIO": "スタジオ",
                    "studio": "スタジオ",
                    "studioo": "スタジオ",
                    "スタUDIO": "スタジオ",
                    " MIRROR": "鏡",
                    "MIRROR": "鏡",
                    "mirror": "鏡",
                    " FLOOR": "床",
                    "FLOOR": "床",
                    "floor": "床",
                    "フLOOR": "床",
                    " LIGHTING": "照明",
                    "LIGHTING": "照明",
                    "lighting": "照明",
                    "LIGHINING": "照明",
                    " BARRE": "バー",
                    "BARRE": "バー",
                    "barre": "バー",
                    "BAR": "バー",
                    "bar": "バー",
                    "PROFESSIONAL": "プロフェッショナル",
                    "professional": "プロフェッショナル",
                    "WIDE AREA": "広々とした空間",
                    "DANCE CLASS": "ダンスクラス",
                }
                for eng, jp in replacements.items():
                    final_text = final_text.replace(eng, jp)

                return final_text.strip()

        return response.strip()

    def _format_qwen_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Qwen2.5のチャット形式にフォーマット"""
        formatted_parts = []
        if system_prompt:
            formatted_parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        formatted_parts.append("<|im_start|>assistant\n")
        return "\n".join(formatted_parts)

    def _format_tinyllama_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """TinyLlamaのチャット形式にフォーマット"""
        formatted_parts = []
        if system_prompt:
            formatted_parts.append(f"<|system|>\n{system_prompt}</s>")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_parts.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                formatted_parts.append(f"<|assistant|>\n{content}</s>")

        formatted_parts.append("<|assistant|>")
        return "\n".join(formatted_parts)

    def _format_phi_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Phi用のシンプルなチャット形式"""
        formatted_parts = []

        if system_prompt:
            formatted_parts.append(f"<|system|>\n{system_prompt}<|end|>\n")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_parts.append(f"<|user|>\n{content}<|end|>\n")
            elif role == "assistant":
                formatted_parts.append(f"<|assistant|>\n{content}<|end|>\n")

        formatted_parts.append("<|assistant|>\n")
        return "".join(formatted_parts)

    def _format_simple_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """日本語モデル用の詳細な形式にフォーマット"""
        formatted_parts = []

        # システムプロンプトを最初に追加
        if system_prompt:
            formatted_parts.append(f"システム: {system_prompt}\n")

        # 会話履歴を追加（最後の6メッセージまで）
        recent_messages = messages[-6:] if len(messages) > 6 else messages

        if len(recent_messages) > 0:
            for msg in recent_messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    formatted_parts.append(f"ユーザー: {content}")
                elif role == "assistant":
                    formatted_parts.append(f"アシスタント: {content}")

        formatted_parts.append("\nアシスタント:")
        return "\n".join(formatted_parts)

    def _format_prompt(
        self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None, model_name: str = None
    ) -> str:
        """
        チャット履歴をプロンプト形式にフォーマット
        モデルに応じた最適な形式を使用

        Args:
            messages: メッセージ履歴のリスト
            system_prompt: システムプロンプト
            model_name: モデル名（フォーマット判定用）

        Returns:
            フォーマットされたプロンプト文字列
        """
        if not model_name:
            model_name = self.model_name

        # モデルに応じたフォーマットを選択
        if "Qwen" in model_name:
            return self._format_qwen_prompt(messages, system_prompt)
        elif "Phi" in model_name or "phi" in model_name:
            return self._format_phi_prompt(messages, system_prompt)
        elif "TinyLlama" in model_name:
            return self._format_tinyllama_prompt(messages, system_prompt)
        else:
            return self._format_simple_prompt(messages, system_prompt)

    def _get_generation_params(self, model_name: str) -> Dict:
        """モデルに応じた最適な生成パラメータを取得"""
        # デフォルトパラメータ
        params = {
            "max_new_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.85,
            "top_k": 40,
            "repetition_penalty": 1.5,
            "no_repeat_ngram_size": 5,
        }

        # モデル固有の調整
        if "Phi" in model_name or "phi" in model_name:
            # Phiはより保守的なパラメータが良い
            params["temperature"] = 0.3
            params["top_p"] = 0.8
            params["max_new_tokens"] = 80
        elif "Qwen" in model_name:
            # Qwen向け
            params["temperature"] = 0.7
            params["top_p"] = 0.85
            params["max_new_tokens"] = 150

        return params

    def _generate_response(self, prompt: str, model_name: str, max_new_tokens: int = 256, task_type: str = "general") -> str:
        """
        LLMを使用して応答を生成

        Args:
            prompt: 入力プロンプト
            model_name: 使用するモデル名
            max_new_tokens: 生成する最大トークン数
            task_type: タスクタイプ（後処理用）

        Returns:
            生成されたテキスト
        """
        # デバッグ用にプロンプトをログ出力
        logger.info(f"=== Prompt for model {model_name} ===\n{prompt}\n=== End Prompt ===")

        # モックモードの場合
        if self.use_mock:
            return self._generate_mock_response(prompt)

        # モデルがロードされていない場合はロード
        if model_name not in self.models:
            self._load_model(model_name)

        try:
            with self.models_lock:
                model = self.models[model_name]
                tokenizer = self.tokenizers[model_name]

            # トークナイズ
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)

            # モデルに応じた生成パラメータを取得
            gen_params = self._get_generation_params(model_name)

            # 生成パラメータの調整
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_params["max_new_tokens"],
                do_sample=True,
                temperature=gen_params["temperature"],
                top_p=gen_params["top_p"],
                top_k=gen_params["top_k"],
                repetition_penalty=gen_params["repetition_penalty"],
                no_repeat_ngram_size=gen_params["no_repeat_ngram_size"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # デコード（入力部分を除外）
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # デバッグ: 生成された生テキストをログ出力
            logger.info(f"=== Raw generated text ===\n{generated_text}\n=== End Raw ===")

            # 特殊トークンやフォーマット記号を削除
            generated_text = (
                generated_text.replace("</s>", "")
                .replace("<|im_start|>", "")
                .replace("<|im_end|>", "")
                .replace("<|assistant|>", "")
                .replace("<|user|>", "")
                .replace("質問:", "")
                .replace("回答:", "")
                .replace("指示:", "")
                .replace("【システムプロンプト】", "")
                .replace("【会話履歴】", "")
                .replace("【応答】", "")
                .replace("ユーザー:", "")
                .replace("アシスタント:", "")
                .strip()
            )

            # 余分な生成を防ぐ - 次のターンが始まったら切り取る
            for delimiter in ["\n<|im_start|>", "\n質問:", "\n回答:", "\n指示:", "\nユーザー:", "\nアシスタント:"]:
                if delimiter in generated_text:
                    generated_text = generated_text.split(delimiter)[0].strip()
                    break

            # 空の応答の場合のフォールバック
            if not generated_text:
                generated_text = "申し訳ありませんが、応答を生成できませんでした。"

            # タスク固有の後処理
            generated_text = self._post_process_response(generated_text, task_type)

            logger.info(f"=== Final response ===\n{generated_text}\n=== End Final ===")

            return generated_text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _generate_mock_response(self, prompt: str) -> str:
        """
        モック応答を生成（開発・デモ用）

        Args:
            prompt: 入力プロンプト

        Returns:
            モック応答
        """
        # プロンプトから最後のユーザーメッセージを抽出
        user_message = ""
        if "<|user|>" in prompt:
            parts = prompt.split("<|user|>")
            if len(parts) > 1:
                last_user = parts[-1].split("</s>")[0].strip()
                user_message = last_user.lower()

        # シンプルなルールベースの応答
        if "こんにちは" in user_message or "hello" in user_message:
            return "こんにちは！どのようにお手伝いできますか？"
        elif "ありがとう" in user_message or "thank" in user_message:
            return "どういたしまして！他に何かお手伝いできることはありますか？"
        elif "さようなら" in user_message or "bye" in user_message:
            return "さようなら！また何かあればお気軽にどうぞ。"
        elif "天気" in user_message or "weather" in user_message:
            return "申し訳ありませんが、私は天気情報にアクセスできません。お近くの気象情報をご確認ください。"
        elif "名前" in user_message or "name" in user_message:
            return "私はOpenVINO AI Toolkitのアシスタントです。"
        elif "?" in user_message or "？" in user_message:
            return f"「{user_message[:50]}」についてのご質問ですね。申し訳ありませんが、現在はモックモードで動作しているため、詳細な回答は提供できません。"
        else:
            return "ご質問ありがとうございます。現在はモックモードで動作しているため、限定的な応答のみ提供しています。実際のLLMモデルを使用する場合は、適切な日本語対応モデル（rinna/japanese-gpt-neox-small等）の設定をご検討ください。"

    def _cleanup_old_sessions(self):
        """古いセッションをクリーンアップ"""
        with self.sessions_lock:
            current_time = datetime.now()
            expired_sessions = []

            for session_id, session in self.sessions.items():
                last_access = session.get("last_access", session["created_at"])
                if current_time - last_access > self.session_timeout:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                del self.sessions[session_id]
                logger.info(f"Cleaned up expired session: {session_id}")

            # セッション数が最大値を超えている場合、最も古いセッションを削除
            if len(self.sessions) > self.max_sessions:
                # 最終アクセス時刻でソート
                sorted_sessions = sorted(self.sessions.items(), key=lambda x: x[1].get("last_access", x[1]["created_at"]))
                # 古いセッションを削除
                num_to_remove = len(self.sessions) - self.max_sessions
                for i in range(num_to_remove):
                    session_id = sorted_sessions[i][0]
                    del self.sessions[session_id]
                    logger.info(f"Removed old session due to limit: {session_id}")

    def _get_or_create_session(
        self,
        session_id: Optional[str],
        task_type: str,
        system_prompt: Optional[str],
        model_name: str,
    ) -> tuple[str, Dict]:
        """セッションを取得または新規作成"""
        if session_id is None or session_id not in self.sessions:
            session_id = str(uuid.uuid4())
            final_system_prompt = self._get_system_prompt_for_task(task_type, system_prompt, model_name)
            self.sessions[session_id] = {
                "messages": [],
                "system_prompt": final_system_prompt,
                "model_name": model_name,
                "task_type": task_type,
                "created_at": datetime.now(),
                "last_access": datetime.now(),
            }
        elif system_prompt:
            self.sessions[session_id]["system_prompt"] = system_prompt
            self.sessions[session_id]["task_type"] = task_type

        if model_name:
            self.sessions[session_id]["model_name"] = model_name

        self.sessions[session_id]["last_access"] = datetime.now()
        return session_id, self.sessions[session_id]

    def _add_user_message(self, session: Dict, message: str) -> None:
        """ユーザーメッセージをセッションに追加"""
        user_message = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        }
        session["messages"].append(user_message)

        if len(session["messages"]) > self.max_history_messages * 2:
            session["messages"] = session["messages"][-(self.max_history_messages * 2) :]

    def _process_task_specific_response(self, response_text: str, task_type: str, message: str) -> str:
        """タスク固有の後処理を実行"""
        if task_type == "image_prompt_improvement":
            return process_prompt_improvement_request(message, response_text)
        elif task_type == "image_prompt_generation":
            return process_prompt_generation_request(message, response_text)
        return response_text

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Dict:
        """
        チャットメッセージを処理して応答を生成

        Args:
            message: ユーザーメッセージ
            session_id: セッションID（指定しない場合は新規作成）
            system_prompt: システムプロンプト（優先）
            model_name: 使用するモデル名（指定しない場合はデフォルト）
            task_type: タスクタイプ（自動検出される、明示的に指定も可）

        Returns:
            応答情報を含む辞書
        """
        if model_name is None:
            model_name = self.model_name

        if task_type is None:
            task_type = self._detect_task_type(message)

        try:
            self._cleanup_old_sessions()

            with self.sessions_lock:
                session_id, session = self._get_or_create_session(session_id, task_type, system_prompt, model_name)
                used_model = session["model_name"]
                self._add_user_message(session, message)
                prompt = self._format_prompt(session["messages"][:], session["system_prompt"], model_name=used_model)

            response_text = self._generate_response(prompt, used_model, task_type=session.get("task_type", "general"))
            response_text = self._process_task_specific_response(response_text, session.get("task_type"), message)

            with self.sessions_lock:
                assistant_message = {
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now().isoformat(),
                }
                session["messages"].append(assistant_message)

            return {
                "response": response_text,
                "session_id": session_id,
                "model": used_model,
                "timestamp": assistant_message["timestamp"],
            }

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {"error": str(e)}

    def get_history(self, session_id: str) -> Dict:
        """
        セッションの会話履歴を取得

        Args:
            session_id: セッションID

        Returns:
            履歴情報を含む辞書
        """
        with self.sessions_lock:
            if session_id not in self.sessions:
                return {"error": "Session not found", "session_id": session_id}

            session = self.sessions[session_id]
            # 最終アクセス時刻を更新
            session["last_access"] = datetime.now()

            return {
                "session_id": session_id,
                "messages": session["messages"][:],  # コピーを返す
                "system_prompt": session["system_prompt"],
                "created_at": session["created_at"].isoformat(),
            }

    def delete_history(self, session_id: str) -> Dict:
        """
        セッションの会話履歴を削除

        Args:
            session_id: セッションID

        Returns:
            削除結果を含む辞書
        """
        with self.sessions_lock:
            if session_id not in self.sessions:
                return {"error": "Session not found", "session_id": session_id}

            del self.sessions[session_id]
            return {"success": True, "session_id": session_id, "message": "History deleted"}

    def list_sessions(self) -> Dict:
        """
        すべてのアクティブセッションをリスト

        Returns:
            セッションリスト
        """
        with self.sessions_lock:
            sessions_info = []
            # 辞書のコピーを作成してから反復
            sessions_copy = dict(self.sessions)

            for sid, session in sessions_copy.items():
                sessions_info.append(
                    {
                        "session_id": sid,
                        "message_count": len(session["messages"]),
                        "created_at": session["created_at"].isoformat(),
                        "last_access": session.get("last_access", session["created_at"]).isoformat(),
                    }
                )

            return {"sessions": sessions_info, "total": len(sessions_info)}
