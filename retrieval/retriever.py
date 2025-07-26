from .vector_store import load_vector_store

class DocumentRetriever:
    def __init__(self, index_path="data/index"):
        self.vectorstore = load_vector_store(index_path)

    def retrieve(self, query: str, k: int = 5):
        """
        Retrieves top-k relevant documents for the given query.
        """
        return self.vectorstore.similarity_search(query, k=k)

def get_retriever(index_path="data/index"):
    return DocumentRetriever(index_path)
