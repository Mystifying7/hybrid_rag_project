# src/ingestion.py
import os
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def process_markdown_file(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Reads a Markdown file, splits it by headers to retain logical structure, 
    and then applies recursive character splitting for uniform chunk sizes.
    """
    # 1. Read the raw document
    with open(file_path, 'r', encoding='utf-8') as f:
        doc_text = f.read()

    # 2. Define the Markdown headers we want to split on.
    # This maps the markdown symbol to a metadata key we can reference later.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Pass 1: Split structurally by Markdown headers
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Keep the header text in the chunk for context
    )
    md_header_splits = markdown_splitter.split_text(doc_text)

    # Pass 2: Split physically by character limits
    # chunk_size defines the max length of a chunk.
    # chunk_overlap ensures context isn't lost at the boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""] # Tries to split by paragraph first, then line, then word
    )
    
    final_chunks = text_splitter.split_documents(md_header_splits)
    
    # Inject source file metadata into each chunk
    file_name = os.path.basename(file_path)
    for chunk in final_chunks:
        chunk.metadata["source"] = file_name
        
    return final_chunks

if __name__ == "__main__":
    # --- Local Testing Block ---
    # Create a dummy markdown file to test our pipeline
    test_dir = "../data/sample_docs"
    os.makedirs(test_dir, exist_ok=True)
    test_file = os.path.join(test_dir, "test_doc.md")
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("""# System Architecture\nWelcome to the architecture guide.\n\n## Database Setup\nWe use PostgreSQL for relational data. Make sure to configure your connection strings securely. Do not hardcode passwords.\n\n### Migration Steps\n1. Run `alembic upgrade head`\n2. Verify the schema in PgAdmin.""")
    
    print(f"Processing {test_file}...\n")
    chunks = process_markdown_file(test_file, chunk_size=150, chunk_overlap=20)
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content:\n{chunk.page_content}\n")