"""
OpenVINOを使ったチャットサービス
軽量LLMを使用してチャット機能を実現
"""

from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import logging
import uuid

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatService:
    """OpenVINOベースのチャットサービス"""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        cache_dir: str = "./models/chat_llm",
    ):
        """
        チャットサービスの初期化

        Args:
            model_name: 使用するLLMモデル名
            cache_dir: モデルキャッシュディレクトリ
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # モデルとトークナイザー
        self.model = None
        self.tokenizer = None

        # セッション管理（メモリ内）
        # session_id -> {"messages": [...], "system_prompt": str, "created_at": datetime}
        self.sessions: Dict[str, Dict] = {}

        # モデルをロード
        self._load_model()

    def _load_model(self):
        """LLMモデルをロード"""
        try:
            logger.info(f"Loading chat model: {self.model_name}")
            model_path = self.cache_dir / self.model_name.replace("/", "_")

            # モデルが既にエクスポートされているか確認
            if not model_path.exists():
                logger.info("Exporting chat model to OpenVINO format...")
                self.model = OVModelForCausalLM.from_pretrained(
                    self.model_name, export=True, compile=True
                )
                self.model.save_pretrained(model_path)
            else:
                logger.info("Loading cached OpenVINO chat model...")
                self.model = OVModelForCausalLM.from_pretrained(
                    model_path, compile=True
                )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # パディングトークンの設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Chat model loaded successfully: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading chat model {self.model_name}: {e}")
            raise

    def _format_prompt(
        self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None
    ) -> str:
        """
        チャット履歴をプロンプト形式にフォーマット

        Args:
            messages: メッセージ履歴のリスト
            system_prompt: システムプロンプト

        Returns:
            フォーマットされたプロンプト文字列
        """
        # TinyLlamaのチャット形式に従う
        # 他のモデルを使用する場合は、このフォーマットを調整する必要がある
        formatted_messages = []

        if system_prompt:
            formatted_messages.append(f"<|system|>\n{system_prompt}</s>")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                formatted_messages.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                formatted_messages.append(f"<|assistant|>\n{content}</s>")

        # 最後にアシスタントの応答を促すプレフィックスを追加
        formatted_messages.append("<|assistant|>\n")

        return "\n".join(formatted_messages)

    def _generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        LLMを使用して応答を生成

        Args:
            prompt: 入力プロンプト
            max_new_tokens: 生成する最大トークン数

        Returns:
            生成されたテキスト
        """
        try:
            # トークナイズ
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )

            # 生成
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # デコード（入力部分を除外）
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        チャットメッセージを処理して応答を生成

        Args:
            message: ユーザーメッセージ
            session_id: セッションID（指定しない場合は新規作成）
            system_prompt: システムプロンプト

        Returns:
            応答情報を含む辞書
        """
        try:
            # セッションIDの処理
            if session_id is None or session_id not in self.sessions:
                session_id = str(uuid.uuid4())
                self.sessions[session_id] = {
                    "messages": [],
                    "system_prompt": system_prompt
                    or "You are a helpful assistant.",
                    "created_at": datetime.now(),
                }
            elif system_prompt:
                # 既存セッションのシステムプロンプトを更新
                self.sessions[session_id]["system_prompt"] = system_prompt

            session = self.sessions[session_id]

            # ユーザーメッセージを追加
            user_message = {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat(),
            }
            session["messages"].append(user_message)

            # プロンプトをフォーマット
            prompt = self._format_prompt(
                session["messages"], session["system_prompt"]
            )

            # 応答を生成
            response_text = self._generate_response(prompt)

            # アシスタントの応答を履歴に追加
            assistant_message = {
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat(),
            }
            session["messages"].append(assistant_message)

            return {
                "response": response_text,
                "session_id": session_id,
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
        if session_id not in self.sessions:
            return {"error": "Session not found", "session_id": session_id}

        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "messages": session["messages"],
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
        sessions_info = []
        for sid, session in self.sessions.items():
            sessions_info.append(
                {
                    "session_id": sid,
                    "message_count": len(session["messages"]),
                    "created_at": session["created_at"].isoformat(),
                }
            )

        return {"sessions": sessions_info, "total": len(sessions_info)}
