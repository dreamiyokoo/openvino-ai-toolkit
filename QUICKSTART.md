# クイックスタートガイド

## 1. 依存パッケージのインストール

```bash
cd /home/yokoo/www/open-vino-trunslate-api
pip install -r requirements.txt
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
2. 翻訳したいテキストを入力
3. ターゲット言語を選択
4. 「翻訳する」ボタンをクリック

### API経由で使用

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

```javascript
// JavaScriptの例
fetch('http://localhost:8000/api/translate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Hello, world!',
    target_lang: 'ja'
  })
})
.then(response => response.json())
.then(data => console.log(data.translated_text));
```

## 4. APIドキュメント

FastAPIの自動生成ドキュメントにアクセスできます:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 5. 注意事項

### 初回起動時

初回起動時、翻訳モデルをHugging Faceからダウンロードするため、
時間がかかる場合があります。モデルは`models/`ディレクトリに
キャッシュされ、2回目以降は高速に起動します。

### メモリ使用量

複数の言語ペアを使用する場合、メモリを多く消費します。
最低4GB以上のメモリを推奨します。

### サポートされている言語

- English (en)
- Japanese (ja)
- Chinese (zh)
- Korean (ko)
- French (fr)
- German (de)
- Spanish (es)
- Russian (ru)

直接のペアがない場合、英語を経由して翻訳します。

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
