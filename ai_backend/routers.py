from fastapi import APIRouter, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from ai_backend.service import CommentService

router = APIRouter(prefix="/tonality", tags=["tonality"])

def get_service() -> CommentService:
    return CommentService()

@router.post("/text")
async def analyze_text(
    text: str = Form(...),
    service: CommentService = Depends(get_service)
):
    tonal = service.analyze_text(text)
    return {"text": text, "tonalnost": tonal}

@router.post("/file")
async def analyze_file(
    file: UploadFile = File(...),
    service: CommentService = Depends(get_service)
):
    content = await file.read()
    df = service.analyze_file(content)
    return df.to_dict(orient="records")
