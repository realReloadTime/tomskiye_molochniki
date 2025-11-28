import pandas as pd
from io import BytesIO
from repository import CommentRepository
from schemas import CommentCreateSchema


def ai(text: str) -> str:
    t = text.lower()
    if "good" in t:
        return "positive"
    if "bad" in t:
        return "negative"
    return "neutral"

class CommentService:
    def __init__(self, repo: CommentRepository):
        self.repo = repo

    def analyze_text(self, text: str) -> str:
        tonal = ai(text)

        self.repo.save_comment(
            CommentCreateSchema(text=text, tonalnost=tonal)
        )

        return tonal

    def analyze_file(self, file_bytes: bytes):
        df = pd.read_csv(BytesIO(file_bytes))

        if "comment" not in df.columns:
            raise Exception("comment один столбик")

        df["tonalnost"] = df["comment"].apply(ai)


        for _, row in df.iterrows():
            self.repo.save_comment(
                CommentCreateSchema(
                    text=row["comment"],
                    tonalnost=row["tonalnost"]
                )
            )

        return df
