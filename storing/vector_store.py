from app.config import get_embedding_model
from langchain.vectorstores import FAISS


def save_vector_store(chunks, path="data/index"):
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(path)

def load_vector_store(path="data/index"):
    embeddings = get_embedding_model()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
