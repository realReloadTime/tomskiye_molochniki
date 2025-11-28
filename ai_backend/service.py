from ai.ai_output import load_toxicity_model, process_toxicity_csv, predict_toxicity_with_probability
from ai_backend.schemas import CommentSchema
from datetime import datetime, UTC

model, vectorizer = load_toxicity_model()


class CommentService:
    async def analyze_text(self, text: str) -> CommentSchema:
        comment, class_label, probability = predict_toxicity_with_probability(text, model, vectorizer)
        return CommentSchema(comment=comment,
                             class_label=class_label,
                             probability=probability,
                             created_date=datetime.now(UTC))

    async def analyze_file(self, file_bytes: bytes):
        return process_toxicity_csv(file_bytes, model, vectorizer)
