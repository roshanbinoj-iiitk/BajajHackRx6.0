from pydantic import BaseModel
from typing import List,Optional

class QueryRequest(BaseModel):
    question:List[str]
    documents:str

class Clause(BaseModel):
    clause_id: str
    text: str


class QueryResponse(BaseModel):
    answer:str
    matched_clauses:List[Clause]

