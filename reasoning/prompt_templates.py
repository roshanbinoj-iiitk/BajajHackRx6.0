from langchain.prompts import ChatPromptTemplate


final_prompt = ChatPromptTemplate.from_template("""
You are an expert assistant for analyzing insurance policies.

Context from the policy document:
{context}

Question:
{question}

Instructions:
- Answer the question using ONLY the information from the context above. Do not add the citation doc number.
- Be concise, precise and give short answer.
- If the context is insufficient, respond with: "The context does not provide enough information to answer this question."
- Do NOT guess or use external knowledge.
- Format the response as plain text.
""")




hyde_prompt = ChatPromptTemplate.from_template("""
Generate a dummy response to the following question the context is not available.
The question is regarding a policy document, be specific to it.
The response should be a dummy response, which will be used for Hyde retrieval.
{question}

""")


generate_query_prompt = ChatPromptTemplate.from_template("""
Generate a better search query for the following user question. The question is regarding a health policy document, be specific to it. 
The response should be plain text , as it will be used for retreival from the policy docs.
Question: {question}
""")

