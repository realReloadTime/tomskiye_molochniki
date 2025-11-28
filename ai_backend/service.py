import pandas as pd
from io import BytesIO
from repository import CommentRepository
from schemas import CommentCreateSchema
from ai.ai_output import load_toxicity_model, process_toxicity_csv, predict_toxicity_with_probability


model, vectorizer = load_toxicity_model()


class CommentService:
    async def analyze_text(self, text: str) -> str:
        tonal = predict_toxicity_with_probability(text, model, vectorizer)
        return tonal

    async def analyze_file(self, file_bytes: bytes):
        return process_toxicity_csv(file_bytes, model, vectorizer)
