import time
from typing import List
from fastapi import APIRouter, HTTPException, Header
from app.config import get_vector_store
from app.schemas import QueryRequest, QueryResponse
from reasoning.llm_worker import get_answer
from pydantic import BaseModel
import asyncio

from pathlib import Path
from storing.loader import load_documents_from_folder,download_pdf
from storing.chunker import split_documents
from storing.vector_store import save_vector_store



router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        answer = get_answer(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


class BulkQuestionRequest(BaseModel):
    documents:str
    questions: List[str]

class BulkAnswerResponse(BaseModel):
    answers: List[str]

@router.post("/bulk_ask",response_model=BulkAnswerResponse)
async def bulk_ask_questions(request: BulkQuestionRequest):
    async def get_async_answer(question:str)->str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_answer, question)


    download_and_ingest(request.documents)
    index =get_vector_store()[0]
    wait_for_pinecone(index)
    answers = await asyncio.gather(*[get_async_answer(q) for q in request.questions])
    index.delete_namespace(namespace="__default__")  # Clear the index after processing

    return {"answers": answers}





def run_ingestion():
    print("[*] Loading documents...")
    documents = load_documents_from_folder()
    print(f"[*] Loaded {len(documents)} documents.")

    with open("data/raw_documents.txt","w") as f:
        for doc in documents:
            f.write(doc.page_content + "\n\n")
        print("[*] Raw documents saved to data/raw_documents.txt")

    print("[*] Splitting documents...")
    chunks = split_documents(documents)
    print(f"[*] Split into {len(chunks)} chunks.")

    print("[*] Saving to vector store...")
    save_vector_store(chunks,namespace="__default__")
    print("[+] Done. Vector store saved to /data/index.")
    


def download_and_ingest(url:str):
    
    download_pdf(url, Path("data/docs/document.pdf"))
    run_ingestion()





def wait_for_pinecone(index, timeout=30, interval=2):
    """Wait until vectors are available in Pinecone"""
    start = time.time()
    while time.time() - start < timeout:
        stats = index.describe_index_stats()
        vector_count = stats.get("total_vector_count", 0)

        if vector_count > 0:
            print(f"[*] Pinecone is ready with {vector_count} vectors.")
            return
        print("[*] Waiting for Pinecone to be ready...")
        time.sleep(interval)
    
    raise TimeoutError("Vectors not available in Pinecone after timeout.")