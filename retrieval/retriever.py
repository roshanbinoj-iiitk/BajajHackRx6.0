from langchain_core.runnables import Runnable
from typing import Optional, Any
from storing.vector_store import load_vector_store

class DocumentRetriever(Runnable):
    def __init__(self, index_path="data/index", top_k: int = 5):
        self.vectorstore = load_vector_store(index_path)
        self.top_k = top_k

    def invoke(self, input: Any, config: Optional[dict] = None) -> Any:
        """
        Required by Runnable: runs when .invoke() is called or when used in a LangChain chain.
        """
        return self.vectorstore.invoke(str(input), config)


def get_retriever(index_path="data/index", top_k=5):
    return DocumentRetriever(index_path=index_path, top_k=top_k)

