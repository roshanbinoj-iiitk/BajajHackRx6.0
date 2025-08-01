from typing import List
from langchain_core.documents import Document
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
import requests
import os
import shutil

folder_path = "data/docs"

def load_documents_from_folder() -> List[Document]:
    """Load documents from data/docs and return list of LangChain Documents"""
    docs = []
    for file in Path(folder_path).glob("*.pdf"):
        loader = PyPDFLoader(str(file))
        docs.extend(loader.load())
    
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    
    return docs




def download_pdf(url: str, dest_path: Path):
    """Download a PDF from a URL and save it to the specified path."""
    print(f"Downloading PDF from {url} to {dest_path}")
    
    response = requests.get(url)
    response.raise_for_status()  
    with open(dest_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded PDF to {dest_path}")


