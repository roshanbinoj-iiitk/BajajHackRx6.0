from app.config import get_llm
from langchain.prompts import ChatPromptTemplate

from retrieval.retriever import get_retriever
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


llm = get_llm()
prompt = ChatPromptTemplate.from_template("""
You are an expert assistant for analyzing insurance policies and contractual documents.

Context:
{context}

Question:
{question}

Instructions:
- Use only the information provided in the context.
- Be concise and precise.
- If the context is insufficient to answer, respond with: "The context does not provide enough information to answer this question."
- Do NOT guess, add explanations, or use external knowledge.
- Format response as plain text only (no markdown, no bullet points).
""")


hyde_prompt = ChatPromptTemplate.from_template("""
Generate a hypothetical paragraph that answers the following insurance policy question, assuming context is not available.

Question:
{question}

The answer should resemble actual policy text — formal, factual, and specific.
""")


retriever = get_retriever()

generate_query_prompt = ChatPromptTemplate.from_template("""
Generate a better search query for the following user question. The question is regarding a health policy document, be specific to it. 
The response should be plain text , as it will be used for retreival from the policy docs.
Question: {question}
""")


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




final_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. 
The question is regarding a health policy document, contract or email, be specific to it.
                                                

Give a concise answer be specific to the policy document.
                                                
Question:
{question}
""")


another_rag_chain = (
    retrieval_chain
    | prompt
    | llm
    | StrOutputParser()
)

hyde_rag_chain = (
    hyde_retrieval_chain
    | prompt
    | llm
    | StrOutputParser() 
)
    

old_rag_chain = (
    {"context": retriever, "question": RunnableLambda(lambda x: x)}
    | prompt
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

    