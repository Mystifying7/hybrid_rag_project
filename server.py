# server.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import our modular hybrid RAG pipeline components
from src.ingestion import process_markdown_file
from src.indexer import DualIndexer
from src.retriever import HybridRetriever
from src.generator import RAGGenerator

app = FastAPI(title="Smart Documentation Search Engine API")

# Initialize Pipeline globally on startup to prevent reload latency
print("Initializing Hybrid RAG Pipeline...")
docs_path = "data/sample_docs"
all_chunks = []
if os.path.exists(docs_path):
    for root, _, files in os.walk(docs_path):
        for file in files:
            if file.endswith(".md") or file.endswith(".txt"):
                chunks = process_markdown_file(os.path.join(root, file), chunk_size=300, chunk_overlap=40)
                all_chunks.extend(chunks)

indexer = DualIndexer()
if all_chunks:
    indexer.fit(all_chunks)

retriever = HybridRetriever(indexer=indexer)
generator = RAGGenerator(model_name="openai/gpt-oss-120b", temperature=0.0)
print("Pipeline initialized successfully!")

class QueryRequest(BaseModel):
    query: str
    initial_k: int = 6
    final_k: int = 2

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.post("/api/search")
def search_docs(req: QueryRequest):
    if not all_chunks:
        raise HTTPException(status_code=400, detail="No documents indexed in data directory.")
    
    # 1. Retrieve & Rerank via Hybrid Pipeline
    retrieved_docs = retriever.retrieve_and_rerank(req.query, initial_k=req.initial_k, final_k=req.final_k)
    
    # 2. Synthesize Grounded Response
    response = generator.answer_query(req.query, retrieved_docs)
    
    # 3. Format Context for UI Inspection Panel
    formatted_chunks = []
    for doc in retrieved_docs:
        chunk = doc['chunk']
        formatted_chunks.append({
            "source": chunk.metadata.get('source', 'Unknown'),
            "section": chunk.metadata.get('Header 1', chunk.metadata.get('Header 2', 'N/A')),
            "rrf_score": round(doc.get('rrf_score', 0.0), 5),
            "cross_encoder_score": round(doc.get('cross_encoder_score', 0.0), 4),
            "content": chunk.page_content
        })
        
    return {
        "answer": response["answer"],
        "sources": response["sources"],
        "retrieved_context": formatted_chunks
    }

# Mount static folder for client-side assets
os.makedirs("static", exist_ok=True)