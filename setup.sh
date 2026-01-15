#!/bin/bash

# OpenVINO Translation API セットアップスクリプト

echo "========================================="
echo "OpenVINO Translation API セットアップ"
echo "========================================="
echo ""

# Python3がインストールされているか確認
if ! command -v python3 &> /dev/null; then
    echo "❌ エラー: python3が見つかりません"
    echo "sudo apt install python3 python3-venv を実行してください"
    exit 1
fi

echo "✓ Python3が見つかりました"
python3 --version

# 仮想環境の作成
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Python仮想環境を作成しています..."
    python3 -m venv venv
    echo "✓ 仮想環境を作成しました"
else
    echo "✓ 仮想環境は既に存在します"
fi

# 仮想環境をアクティブ化
echo ""
echo "🔧 仮想環境をアクティブ化しています..."
source venv/bin/activate

# pipをアップグレード
echo ""
echo "📦 pipをアップグレードしています..."
pip install --upgrade pip

# sentencepieceを先にインストール
echo ""
echo "📦 sentencepieceをインストールしています..."
pip install sentencepiece --no-cache-dir

# 依存パッケージをインストール
echo ""
echo "📦 依存パッケージをインストールしています..."
echo "⚠️  これには時間がかかる場合があります（初回は10-30分程度）"
pip install -r requirements.txt

# インストール結果を確認
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ セットアップが完了しました！"
    echo "========================================="
    echo ""
    echo "次のコマンドでアプリケーションを起動できます:"
    echo ""
    echo "  source venv/bin/activate"
    echo "  python main.py"
    echo ""
    echo "または、VS CodeでF5キーを押してデバッグ起動してください。"
    echo ""
else
    echo ""
    echo "========================================="
    echo "❌ インストール中にエラーが発生しました"
    echo "========================================="
    echo ""
    echo "エラーログを確認してください。"
    exit 1
fi
