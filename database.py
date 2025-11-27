from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import DeclarativeBase

SQL_DB_URL = "postgresql://postgres:admin111@localhost:5432/sentiment_db"


engine = create_engine(
    url=SQL_DB_URL,
    echo=False  
)

def get_engine():
    return engine
session_local = sessionmaker(
    autoflush=False,
    expire_on_commit=False,
    bind=engine
)

def get_db():
    db = session_local()
    try:
        yield db
    finally:
        db.close()

class Base(DeclarativeBase):
    pass
