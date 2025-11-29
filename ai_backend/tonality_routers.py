from fastapi import APIRouter, UploadFile, File, Form, Depends
from service import CommentService  # ← прямой импорт
from schemas import CommentSchema

router = APIRouter(prefix="/tonality", tags=["tonality"])

def get_service() -> CommentService:
    return CommentService()

@router.post("/text", response_model=CommentSchema)
async def analyze_text(
    text: str = Form(...),
    service: CommentService = Depends(get_service)
):
    return await service.analyze_text(text)

@router.post("/file")
async def analyze_file(
    file: UploadFile = File(...),
    service: CommentService = Depends(get_service)
):
    return await service.analyze_file(await file.read())
