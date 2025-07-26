# file: storing/vector_store.py

from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from app.config import get_embedding_model

def save_vector_store(chunks, path="data/index"):
    embeddings = get_embedding_model()

    print("[*] Building vector store with progress:")

    # Show progress as chunks are processed
    documents = []
    for chunk in tqdm(chunks, desc="Embedding chunks", unit="chunk"):
        documents.append(chunk)

    # Build and persist the Chroma vector store
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=path)

    print(f"[+] Vector store saved to {path}")


def load_vector_store(path="data/index"):
    embeddings = get_embedding_model()
    chroma = Chroma(persist_directory=path, embedding_function=embeddings)
    return chroma.as_retriever()