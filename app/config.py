from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
# app/config.py
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

class Settings(BaseSettings):
    gemini_api_key: str
    vector_store_path: str = "./data/embeddings"
    embedding_model_name: str = "models/embedding-001"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

@lru_cache()
def get_llm():
    settings = get_settings()
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=settings.gemini_api_key)

@lru_cache()
def get_embedding_model():
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model='gemini-embedding-001',
        google_api_key=settings.gemini_api_key
    )
