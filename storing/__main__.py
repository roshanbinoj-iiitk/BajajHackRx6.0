from pathlib import Path
from storing.loader import load_documents_from_folder,download_pdf
from storing.chunker import split_documents
from storing.vector_store import save_vector_store

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

    with open("data/chunks.txt","w") as f:
        for chunk in chunks:
            f.write(chunk.page_content + "\n\n")
        print("[*] Chunks saved to data/chunks.txt")

    # print("[*] Saving to vector store...")
    # save_vector_store(chunks)
    # print("[+] Done. Vector store saved to /data/index.")
    


def download_and_ingest(url:str):
    
    download_pdf(url, Path("data/docs/document.pdf"))
    run_ingestion()

if __name__ == "__main__":
    run_ingestion()
