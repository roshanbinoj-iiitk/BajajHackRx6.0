# file: storing/vector_store.py
from langchain_core.documents import Document

from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from app.config import get_embedding_model, get_index
from langchain_pinecone import PineconeVectorStore

import json

BATCH_SIZE = 96

def clean_metadata(meta):
    try:
        return json.loads(json.dumps(meta))  # strip unserializables
    except Exception:
        return {}

def prepare_pinecone_data(docs: list[Document]):
    return [
        {
            "id": f"doc-{i}",
            "text": doc.page_content,
            **clean_metadata(doc.metadata)
        }
        for i, doc in enumerate(docs)
    ]

def save_vector_store(docs: list[Document]):
    index = get_index()

    all_data = prepare_pinecone_data(docs)

    for i in range(0, len(all_data), BATCH_SIZE):
        batch = all_data[i:i + BATCH_SIZE]
        index.upsert_records("__default__",batch)
        print(f"Upserted batch {i // BATCH_SIZE + 1} with {len(batch)} items")



# def load_vector_store(path="data/index"):
#     embeddings = get_embedding_model()
#     chroma = Chroma(persist_directory=path, embedding_function=embeddings)
#     return chroma.as_retriever()





