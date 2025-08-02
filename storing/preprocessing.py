import pdfplumber
import re
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, '..', 'data/docs', 'policy.pdf')


def tables_to_markdown(page, table_settings={}):
    """
    Extracts tables from a pdfplumber page and converts them to Markdown format.
    """
    tables = page.find_tables(table_settings)
    if not tables:
        return ""

    markdown_tables = []
    for table in tables:
        # Extract table data, handling None values
        data = table.extract()
        if not data:
            continue
            
        header = " | ".join(str(h) for h in data[0])
        separator = " | ".join(["---"] * len(data[0]))
        body = "\n".join([" | ".join(str(c) for c in row) for row in data[1:]])
        
        markdown_tables.append(f"| {header} |\n| {separator} |\n{body}")
        
    return "\n\n".join(markdown_tables)

def clean_and_extract_text(page):
    """
    """
    text = page.extract_text(x_tolerance=2, y_tolerance=2)
    if not text:
        return ""
    
    # Optional: Remove extra newlines to make the text more compact
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', text.strip().lower())
    
    return cleaned_text
    
def process_pdf_for_chunking(pdf_path: str):
    """
    Main function to process a PDF.
    1. Extracts and cleans text page by page.
    2. Extracts and formats tables.
    3. Combines all content into a single string.
    """
    all_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract and clean the main text body
            cleaned_text = clean_and_extract_text(page)
            
            # Extract tables and convert them to markdown
            # This helps keep table data structured
            markdown_table = tables_to_markdown(page)
            
            page_content = []
            if cleaned_text:
                page_content.append(cleaned_text)
            if markdown_table:
                # Add a clear separator for context
                page_content.append("\n\n--- TABLE START ---\n\n")
                page_content.append(markdown_table)
                page_content.append("\n\n--- TABLE END ---\n\n")

            if page_content:
                all_content.append("".join(page_content))

    # Join all pages' content into one large string
    # Pages are separated by a clear marker for the text splitter
    full_document_text = "\n\n--- NEW PAGE ---\n\n".join(all_content)
    
    # Wrap the single text string in a LangChain Document object
    # The splitter expects a list of Document objects
    return [Document(page_content=full_document_text)]


def split_documents(document, chunk_size=1000, chunk_overlap=200):
    """
    The final splitting function, now operating on clean, structured data.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""] # More robust separators
    )
    return splitter.split_documents(document)



# --- Main Execution ---
# if __name__ == "__main__":
    pdf_file_path = pdf_path  # Make sure policy.pdf is in the same directory

    # 1. Process the PDF to get clean, structured text in Document format
    print("Step 1: Processing PDF to extract and clean content...")
    processed_docs = process_pdf_for_chunking(pdf_file_path)
    print(f"PDF processed into a single document of length {len(processed_docs[0].page_content)} characters.")

    # 2. Split the clean document into semantic chunks
    print("\nStep 2: Splitting the processed document into chunks...")
    chunks = split_documents(processed_docs)
    print(f"Splitting complete. Created {len(chunks)} chunks.")

    store_chunks_locally(chunks)  # Save chunks to a file

    # 3. Review the output to see the improvement
    print("\n--- Reviewing Improved Chunks ---")
    
    print("\n--- Chunk 1 (Example of clean text) ---")
    print(chunks[0].page_content)   
    
    # Find a chunk that contains a table to demonstrate
    table_chunk_found = False
    for i, chunk in enumerate(chunks):
        if "--- TABLE START ---" in chunk.page_content:
            print(f"\n--- Chunk {i+1} (Example of a structured table) ---")
            print(chunk.page_content)
            table_chunk_found = True
            break
            
    if not table_chunk_found:
        print("\nNo chunk with a complete table found for demonstration (table might be larger than chunk size).")