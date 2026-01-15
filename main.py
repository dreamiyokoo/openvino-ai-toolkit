"""
OpenVINO翻訳API - FastAPIアプリケーション
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import uvicorn
from translation_service import TranslationService

# FastAPIアプリケーションの初期化
app = FastAPI(
    title="OpenVINO Translation API",
    description="OpenVINOを使った高速翻訳APIとWebインターフェース",
    version="1.0.0",
)

# 静的ファイルとテンプレートの設定
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 翻訳サービスの初期化
translation_service = TranslationService()


# リクエストモデル
class TranslationRequest(BaseModel):
    text: str
    target_lang: str
    source_lang: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, world!",
                "target_lang": "ja",
                "source_lang": "en",
            }
        }


# レスポンスモデル
class TranslationResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str
    original_text: str
    via_english: Optional[bool] = False


# ルートページ - Webインターフェース
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """翻訳Webインターフェースを表示"""
    languages = translation_service.get_supported_languages()
    return templates.TemplateResponse(
        "index.html", {"request": request, "languages": languages}
    )


# 翻訳API エンドポイント
@app.post("/api/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    テキストを翻訳するAPIエンドポイント

    - **text**: 翻訳するテキスト
    - **target_lang**: ターゲット言語コード (例: "ja", "en", "zh")
    - **source_lang**: ソース言語コード (オプション)
    """
    result = translation_service.translate(
        text=request.text,
        target_lang=request.target_lang,
        source_lang=request.source_lang,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return TranslationResponse(**result)


# サポートされている言語のリストを取得
@app.get("/api/languages")
async def get_languages():
    """
    サポートされている言語のリストを取得
    """
    return {"languages": translation_service.get_supported_languages()}


# ヘルスチェック
@app.get("/api/health")
async def health_check():
    """APIヘルスチェック"""
    return {"status": "healthy", "service": "OpenVINO Translation API"}


if __name__ == "__main__":
    # アプリケーションを起動
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
