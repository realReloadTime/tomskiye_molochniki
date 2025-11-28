from pydantic import BaseModel
from datetime import datetime

class CommentSchema(BaseModel):
    comment: str
    class_label: float
    probability: float
    created_date: datetime
