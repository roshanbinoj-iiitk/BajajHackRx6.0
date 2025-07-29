from langchain_core.runnables import Runnable
from typing import Optional, Any
from app.config import get_index

class DocumentRetriever(Runnable):
    def __init__(self, index_path:str, top_k: int):
        self.index = get_index()
        self.top_k = top_k

    def invoke(self, input: Any, config: Optional[dict] = None) -> Any:
        """
        Required by Runnable: runs when .invoke() is called or when used in a LangChain chain.
        """
        # print("direct input :",input)
        result = self.index.search(
            namespace="__default__",
            query={
                "inputs":{"text":input['question']},
                "top_k":self.top_k
            }
        )
        print("Retrieval result : ",result)

        return result



def get_retriever(index_path="data/index", top_k=5):
    return DocumentRetriever(index_path=index_path, top_k=top_k)

