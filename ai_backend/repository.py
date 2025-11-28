from sqlalchemy.orm import Session
from models import Comment
from schemas import CommentSchema, CommentCreateSchema

class CommentRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_comment(self, schema: CommentCreateSchema) -> CommentSchema:
        new_comment = Comment(**schema.model_dump())
        self.db.add(new_comment)
        self.db.commit()

        return CommentSchema(
            id=new_comment.id,
            text=new_comment.text,
            tonalnost=new_comment.tonalnost,
            created_date=new_comment.created_date
        )
