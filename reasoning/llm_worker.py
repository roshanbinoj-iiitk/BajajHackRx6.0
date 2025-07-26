from app.config import get_llm
from langchain.prompts import ChatPromptTemplate

from retrieval.retriever import get_retriever
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


llm = get_llm()

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}
""")

retriever = get_retriever()


rag_chain = (
    {"context": retriever, "question": RunnableLambda(lambda x: x)}
    | prompt
    | llm
    | StrOutputParser()
)

def get_answer(question: str):
    """
    Run the RAG chain with the given question.
    """
    return rag_chain.invoke({"question": question})


if __name__ == "__main__":
    question = "what are terms in bajaj insurance?"
    answer = get_answer(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")