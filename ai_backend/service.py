
import io
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai'))

from ai_output import load_toxicity_model, process_toxicity_csv, predict_toxicity_with_probability
from schemas import CommentSchema

from datetime import datetime, UTC
import pandas as pd
import chardet

model, vectorizer = load_toxicity_model()


class CommentService:
    async def analyze_text(self, text: str) -> CommentSchema:
       
        comment, class_label, probability = predict_toxicity_with_probability(text, model, vectorizer)
        
        print(f"DEBUG: Original prediction - prob={probability}, class={class_label}")
       
        if probability < 0.4:
            tone_class = 0  # позитивный
            tone_probability = (0.4 - probability) / 0.4 * 100
        elif probability > 0.6:
            tone_class = 2  # негативный  
            tone_probability = (probability - 0.6) / 0.4 * 100
        else:
            tone_class = 1  # нейтральный
            tone_probability = 50
        
        tone_probability = max(0, min(100, tone_probability))
        
       
        
        return CommentSchema(
            comment=comment,
            class_label=tone_class,  
            probability=tone_probability,
            created_date=datetime.now(UTC)
        )

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