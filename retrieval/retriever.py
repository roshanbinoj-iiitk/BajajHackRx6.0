from langchain_core.runnables import Runnable
from typing import Optional, Any
from app.config import get_vector_store
import time

class DocumentRetriever(Runnable):
    def __init__(self, index_path:str, top_k: int):
        self.index,self.pc = get_vector_store()
        self.top_k = top_k


    def rerank(self,docs:Any,query:str):
        """
        Rerank the results based on the input.
        This method is not used in this implementation but is required by Runnable.
        """


        ranked_results = self.pc.inference.rerank(
            
            model="bge-reranker-v2-m3",
            query = query,
            documents=docs,
            top_n=3,
            rank_fields=["text"],
            return_documents=True,
         parameters = {
             "truncate":"END"
             } 

        )
        return ranked_results.data

        

    def invoke(self, input: Any, config: Optional[dict] = None) -> Any:
        """
        Required by Runnable: runs when .invoke() is called or when used in a LangChain chain.
        """

        print("Hyde retriever invoked with input:", input)
        # print("direct input :",input)
        print("Retrieving ...")
        start_time = time.time()
        result = self.index.search(
            namespace="__default__",
            query={
            "inputs": {"text": input['question']},
            "top_k": self.top_k
            }
        )
        end_time = time.time()

        print("[*] Got the results , reranking them....")
        print(f"Search results: {result['result']['hits']}")
        formatted_docs = [{"_id": doc["_id"], "text": doc["fields"]["text"]} for doc in result["result"]["hits"]]

        reranked_result = self.rerank(formatted_docs, input['question'])

        print("Reranked result : ",reranked_result)
        print(f"Search time: {end_time - start_time:.4f} seconds")

        # context = "\n\n".join([doc["document"]["text"] for doc in reranked_result])

        # print("[*] Returning the results....", context)

        return reranked_result

        # context = "\n\n".join([doc['fields']['text'] for doc in result["result"]["hits"]])

        # print("[*] Returning the results....",context)
        # return context
  



def get_retriever(index_path="data/index", top_k=10):
    return DocumentRetriever(index_path=index_path, top_k=top_k)

