from langchain_unstructured import UnstructuredLoader
from typing import List
from langchain_core.documents import Document
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_documents_from_folder() -> List[Document]:
    """Load documents from data/docs and return list of LangChain Documents"""
    docs = []
    for file in Path("data/docs").glob("*.pdf"):
        loader = PyPDFLoader(str(file))
        docs.extend(loader.load())
    return docs
