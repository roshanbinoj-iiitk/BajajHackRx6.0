from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(document, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(document)
    store_chunks_locally(chunks)
    return chunks


def store_chunks_locally(chunks):
    with open("data/chunkers.txt","w") as f:
        for chunk in chunks:
            f.write(chunk.page_content + "\n\n")
        print("[*] Chunks saved to data/chunks.txt")

