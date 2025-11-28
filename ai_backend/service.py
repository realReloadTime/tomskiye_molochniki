from ai.ai_output import load_toxicity_model, process_toxicity_csv, predict_toxicity_with_probability
from ai_backend.schemas import CommentSchema
from datetime import datetime, UTC
import os

model, vectorizer = load_toxicity_model()


class CommentService:
    async def analyze_text(self, text: str) -> CommentSchema:
        comment, class_label, probability = predict_toxicity_with_probability(text, model, vectorizer)
        return CommentSchema(comment=comment,
                             class_label=class_label,
                             probability=probability,
                             created_date=datetime.now(UTC))

    async def analyze_file(self, file_bytes: bytes):
        df = process_toxicity_csv(file_bytes, model, vectorizer)
        curr_time = datetime.now(UTC)
        df.to_csv(f"temp_response_{curr_time}.csv", index = False)
        new_file_bytes = open(f"temp_response_{curr_time}.csv", mode="rb")
        new_file_bytes.close()
        os.remove(new_file_bytes)
        return new_file_bytes

          