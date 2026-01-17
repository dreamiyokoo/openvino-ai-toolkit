# クイックスタートガイド - OpenVINO AI Toolkit

## 1. 依存パッケージのインストール

```bash
cd /home/yokoo/www/open-vino-trunslate-api
pip install -r requirements.txt
```

**オプション**: LangChain統合を使用する場合
```bash
pip install langchain
```

## 2. アプリケーションの起動方法

### 方法A: VS Codeから起動 (推奨)

1. VS Codeでこのフォルダを開く
2. `F5`キーを押す
3. "FastAPI: OpenVINO Translation"を選択
4. ブラウザで http://localhost:8000 にアクセス

### 方法B: コマンドラインから起動

```bash
python main.py
```

または

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 3. 使い方

### Webインターフェース

1. ブラウザで http://localhost:8000 を開く
2. **翻訳タブ**:
   - 翻訳したいテキストを入力
   - ターゲット言語を選択
   - 「翻訳する」ボタンをクリック
3. **チャットタブ**:
   - 「チャット」タブをクリック
   - メッセージを入力
   - 「送信」ボタンをクリック
   - AIアシスタントと会話

### 翻訳API

```bash
# curlコマンドの例
curl -X POST "http://localhost:8000/api/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "target_lang": "ja"
  }'
```

```python
# Pythonの例
import requests

response = requests.post(
    "http://localhost:8000/api/translate",
    json={
        "text": "Hello, world!",
        "target_lang": "ja"
    }
)

print(response.json()["translated_text"])
```

### チャットAPI

```bash
# curlコマンドの例
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "こんにちは！",
    "system_prompt": "あなたは親切なアシスタントです"
  }'
```

```python
# Pythonの例
import requests

# 新しいチャットセッション
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "message": "こんにちは！",
        "system_prompt": "あなたは親切なアシスタントです"
    }
)

data = response.json()
print(f"Assistant: {data['response']}")
session_id = data['session_id']

# 同じセッションで会話を続ける
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "message": "今日の天気は？",
        "session_id": session_id
    }
)

print(f"Assistant: {response.json()['response']}")
```

### LangChain統合（オプション）

```python
from langchain_adapter import create_langchain_chat

# LangChain互換のチャットインスタンスを作成
chat = create_langchain_chat(
    system_prompt="あなたは親切なアシスタントです"
)

# チャットを実行
response = chat("こんにちは！")
print(response)
```

## 4. APIドキュメント

FastAPIの自動生成ドキュメントにアクセスできます:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 5. 注意事項

### 初回起動時

初回起動時、モデルをHugging Faceからダウンロードするため、
時間がかかる場合があります。

- **翻訳モデル**: `models/`ディレクトリにキャッシュ
- **チャットLLMモデル**: `models/chat_llm/`ディレクトリにキャッシュ

2回目以降は高速に起動します。

### メモリ使用量

- **翻訳のみ**: 4GB以上推奨
- **翻訳 + チャット**: 8GB以上推奨
- チャットLLMモデル（TinyLlama-1.1B）は約2GBのメモリを使用

### 使用モデル

- **翻訳**: Helsinki-NLP/opus-mt シリーズ
- **チャット**: TinyLlama/TinyLlama-1.1B-Chat-v1.0（デフォルト）

チャットモデルは`chat_service.py`で変更可能。

### サポートされている言語

**翻訳**:
- English (en)
- Japanese (ja)
- Chinese (zh)
- Korean (ko)
- French (fr)
- German (de)
- Spanish (es)
- Russian (ru)
- Thai (th)

直接のペアがない場合、英語を経由して翻訳します。

**チャット**:
- デフォルト（TinyLlama）は英語メイン
- 日本語対応が必要な場合は、rinnaやQwenなどのモデルに変更を推奨

## 6. トラブルシューティング

### ポート8000が使用中の場合

main.pyの最後の行を編集してポート番号を変更:

```python
uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
```

### モデルのダウンロードが遅い場合

環境変数を設定:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python main.py
```
