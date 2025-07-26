from fastapi import FastAPI
from app.api import router

app = FastAPI(
    title="HackRx API",
    description="API for HackRx application",
    version="0.0.469"
)

app.include_router(router)

@app.get("/ping")
def health_check():
    return {"status": "ok"}