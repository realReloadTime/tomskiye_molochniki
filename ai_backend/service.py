import io

from ai.ai_output import load_toxicity_model, process_toxicity_csv, predict_toxicity_with_probability
from ai_backend.schemas import CommentSchema
from datetime import datetime, UTC
import os
import pandas as pd
import chardet

model, vectorizer = load_toxicity_model()


class CommentService:
    async def analyze_text(self, text: str) -> CommentSchema:
        comment, class_label, probability = predict_toxicity_with_probability(text, model, vectorizer)
        return CommentSchema(comment=comment,
                             class_label=class_label,
                             probability=probability,
                             created_date=datetime.now(UTC))

    async def analyze_file(self, file_bytes: bytes) -> bytes | None:
        try:
            opened_file = io.BytesIO(file_bytes)
            encoding = chardet.detect(file_bytes)
            got_df = pd.read_csv(opened_file, encoding=encoding['encoding'], delimiter="^$^#$@&*!")

            df = process_toxicity_csv(got_df, model, vectorizer)

            curr_time = str(datetime.now(UTC).timestamp()).replace(".", "-")
            df.to_csv(curr_time + ".csv", index=False)
            df.to_csv(f"temp_response_{curr_time}.csv", index = False)
            new_file_bytes = open(f"temp_response_{curr_time}.csv", mode="rb").read()

            os.remove(f"temp_response_{curr_time}.csv")
            return new_file_bytes

        except Exception:
            print('ai_backend/service.py error in def analyze_file')
            return None
