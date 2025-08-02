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
from storing.preprocessing import process_pdf_for_chunking
from storing.vector_store import save_vector_store



router = APIRouter()

# @router.post("/hackrx/run",response_model=FastResponse)
# async def hackrx_run(
#     req: Requ,
#     authorization: str = Header(...)
# ):
#     print(authorization)
#     # Check for Bearer token
#     # if not authorization or not authorization.startswith("Bearer "):
#     #     raise HTTPException(
#     #         status_code=status.HTTP_401_UNAUTHORIZED,
#     #         detail="Missing or invalid Authorization header"
#     #     )
#     token = authorization.split(" ")[1]
#     # if not req.question.strip():
#     #         raise HTTPException(status_code=400, detail="Query cannot be empty")


#     dummy_answers = [
#         "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
#         "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
#         "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
#         "The policy has a specific waiting period of two (2) years for cataract surgery.",
#         "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
#         "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
#         "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
#         "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
#         "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
#         "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
#     ]
#     # Return as many answers as questions, or dummy answers if not enough
#     answers = dummy_answers[:len(req.questions)]
#     return {"answers": answers}


# @router.post("/get_response", response_model=QueryResponse)
# async def get_response(req: QueryRequest,
#                        authorization: str = Header(...)):
#     print(authorization)
#     # Check for Bearer token
#     if not authorization or not authorization.startswith("Bearer "):
#         raise HTTPException(
#             status_code=401,
#             detail="Missing or invalid Authorization header"
#         )
#     token = authorization.split(" ")[1]
#     if token != "your":
#         raise HTTPException(
#             status_code=403,
#             detail="Invalid API key"
#         )   
    
#     return QueryResponse(
#         answer="Dummy answer based on the provided documents and questions.",
#         matched_clauses=[
#             Clause(clause_id="1", text="Dummy clause text 1"),
#             Clause(clause_id="2", text="Dummy clause text 2")
#         ]
#     )


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




# Older code for ingestion, kept for reference
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
    
def run_new_ingestion():
    print("[*] Processing PDF to extract and clean content...")
    documents = process_pdf_for_chunking(Path("data/docs/document.pdf"))
    print("[*] PDF processing complete.")
    print("[*] Splitting documents...")
    chunks = split_documents(documents)
    print(f"[*] Split into {len(chunks)} chunks.")

    print("[*] Saving to vector store...")
    save_vector_store(chunks,namespace="__default__")
    print("[+] Done. Vector store saved to /data/index.")
        

def download_and_ingest(url:str):
    
    download_pdf(url, Path("data/docs/document.pdf"))
    run_new_ingestion()





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