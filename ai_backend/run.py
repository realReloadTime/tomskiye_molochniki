import uvicorn
from fastapi import FastAPI
from ai_backend.tonality_routers import router

app = FastAPI(title="tonality sentiment")
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("run:app")
