from pydantic import BaseModel
from datetime import datetime

class CommentSchema(BaseModel):
    id: int
    text: str
    tonalnost: str
    created_date: datetime

class CommentCreateSchema(BaseModel):
    text: str
    tonalnost: str
