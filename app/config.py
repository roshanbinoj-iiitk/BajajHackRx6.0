from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec


load_dotenv()
# app/config.py
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_openai import ChatOpenAI

class Settings(BaseSettings):
    gemini_api_key: str
    vector_store_path: str = "./data/embeddings"
    embedding_model_name: str = "models/embedding-001"
    pinecone_api_key:str
    secret_key:str
    openai_api_key: str

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

@lru_cache()
def get_llm():
    settings = get_settings()
    # return ChatOpenAI(
    #     model="gpt-4o",
    #     openai_api_key=settings.openai_api_key,
    #     temperature=0.0

    # )
    return ChatGoogleGenerativeAI(model="gemini-2.5-pro",google_api_key=settings.gemini_api_key,temperature=0.0)

@lru_cache()
def get_embedding_model():
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model='gemini-embedding-001',
        google_api_key=settings.gemini_api_key
    )

@lru_cache()    
def get_vector_store():
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = "hackrx"  # change when required


    # if not pc.has_index(index_name):
    #     pc.create_index(
    #         name=index_name,
    #         dimension=1024,
    #         metric="cosine",
    #         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    #     )

    index = pc.Index(index_name)
    return (index,pc)