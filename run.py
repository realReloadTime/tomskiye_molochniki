import uvicorn
from fastapi import FastAPI
from sentiment_router import router
from database import Base, get_engine

app = FastAPI(title="sentiment")
app.include_router(router)

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=get_engine())

if __name__ == "__main__":
    uvicorn.run("run:app", reload=True)
