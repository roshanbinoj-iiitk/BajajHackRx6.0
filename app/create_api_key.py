# app/api_key.py
import secrets
from fastapi import APIRouter
from pydantic import BaseModel
from app.db import SessionLocal, APIKey

api_key_router = APIRouter()

class APIKeyResponse(BaseModel):
    api_key: str

@api_key_router.post("/generate-api-key", response_model=APIKeyResponse)
def generate_api_key():
    db = SessionLocal()
    key = secrets.token_urlsafe(16)

    db_key = APIKey(key=key)
    db.add(db_key)
    db.commit()
    db.refresh(db_key)
    db.close()

    return {"api_key": key}
