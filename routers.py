from fastapi import APIRouter, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from database import get_db
from service import CommentService
from repository import CommentRepository

router = APIRouter(prefix="/tonalnost", tags=["tonalnost"])

def get_repo(db: Session = Depends(get_db)) -> CommentRepository:
    return CommentRepository(db)

def get_service(repo: CommentRepository = Depends(get_repo)) -> CommentService:
    return CommentService(repo)

@router.post("/analyze-text")
def analyze_text(
    text: str = Form(...),
    service: CommentService = Depends(get_service)
):
    tonal = service.analyze_text(text)
    return {"text": text, "tonalnost": tonal}

@router.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    service: CommentService = Depends(get_service)
):
    content = await file.read()
    df = service.analyze_file(content)
    return df.to_dict(orient="records")
