from app.config import get_llm
from reasoning.prompt_templates import final_prompt,hyde_prompt,generate_query_prompt
from retrieval.retriever import get_retriever
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


llm = get_llm()



retriever = get_retriever()



generate_query_chain = (
    {"question": lambda x: x["question"]}
    | generate_query_prompt
    | llm
    | StrOutputParser()
)


generate_hyde_chain = (
    {"question": lambda x: x["question"]}
    | hyde_prompt
    | llm
    | StrOutputParser()
)

retrieval_chain = (
    {"question": lambda x: x["question"], "better_query": generate_query_chain}
    | RunnableLambda(lambda x: {
        "question": x["question"],
        "context": retriever.invoke({"question": x["better_query"]})
    })
)

hyde_retrieval_chain = (
    {"question": lambda x: x["question"], "hyde_response": generate_hyde_chain}
    | RunnableLambda(lambda x: {
        "question": x["question"],
        "context": retriever.invoke({"question": x["hyde_response"]})
    })
)



another_rag_chain = (
    retrieval_chain
    | final_prompt
    | llm
    | StrOutputParser()
)

hyde_rag_chain = (
    hyde_retrieval_chain
    | final_prompt
    | llm
    | StrOutputParser() 
)
    

old_rag_chain = (
    {"context": retriever, "question": RunnableLambda(lambda x: x)}
    | final_prompt
    | llm
    | StrOutputParser()
)

def get_answer(question: str):
    """
    Run the RAG chain with the given question.
    """
    return hyde_rag_chain.invoke({"question": question})


if __name__ == "__main__":
    question = "what are terms in bajaj insurance?"
    answer = get_answer(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")  

