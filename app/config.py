from pydantic import BaseSettings
from functools import lru_cache
# app/config.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class Settings(BaseSettings):
    openai_api_key: str
    vector_store_path: str = "./data/embeddings"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()



@lru_cache()
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")