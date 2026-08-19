# server.py
import os
import time
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modular hybrid RAG pipeline components
from src.ingestion import process_markdown_file
from src.indexer import DualIndexer
from src.retriever import HybridRetriever
from src.generator import RAGGenerator

app = FastAPI(title="Smart Documentation Search Engine API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_PATH = "data/sample_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_MODEL = "openai/gpt-oss-120b"

all_chunks = []
doc_files_info = []
indexer = None
retriever = None
generator = None

def load_and_index_pipeline():
    global all_chunks, doc_files_info, indexer, retriever, generator
    print("Indexing documents from:", DOCS_PATH)
    all_chunks = []
    doc_files_info = []
    
    if os.path.exists(DOCS_PATH):
        for root, _, files in os.walk(DOCS_PATH):
            for file in files:
                if file.endswith(".md") or file.endswith(".txt"):
                    filepath = os.path.join(root, file)
                    file_size = os.path.getsize(filepath)
                    chunks = process_markdown_file(filepath, chunk_size=300, chunk_overlap=40)
                    all_chunks.extend(chunks)
                    doc_files_info.append({
                        "filename": file,
                        "path": filepath.replace("\\", "/"),
                        "size_bytes": file_size,
                        "chunks_count": len(chunks)
                    })
    
    indexer = DualIndexer(embedding_model_name=EMBEDDING_MODEL)
    if all_chunks:
        indexer.fit(all_chunks)
    
    retriever = HybridRetriever(indexer=indexer, cross_encoder_model=RERANKER_MODEL)
    generator = RAGGenerator(model_name=GENERATOR_MODEL, temperature=0.0)
    print(f"Pipeline ready! Loaded {len(doc_files_info)} document(s) and {len(all_chunks)} chunks.")

# Initialize on startup
load_and_index_pipeline()

class QueryRequest(BaseModel):
    query: str
    initial_k: int = 6
    final_k: int = 2

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.get("/api/stats")
def get_stats():
    return {
        "status": "ready" if all_chunks else "empty",
        "docs_count": len(doc_files_info),
        "chunks_count": len(all_chunks),
        "documents": doc_files_info,
        "models": {
            "embedding": EMBEDDING_MODEL,
            "sparse": "BM25 (Okapi)",
            "reranker": RERANKER_MODEL,
            "generator": GENERATOR_MODEL
        },
        "sample_queries": [
            "How do I rotate authentication secrets?",
            "What database do we use and how do I execute migrations?",
            "What does ERR_DB_TIMEOUT_504 indicate?",
            "What are the error codes for rate limiting and auth?",
            "How do I configure Redis cache?"
        ]
    }

@app.post("/api/reindex")
def reindex_documents():
    try:
        t0 = time.time()
        load_and_index_pipeline()
        duration = round((time.time() - t0) * 1000)
        return {
            "status": "success",
            "message": f"Successfully re-indexed {len(all_chunks)} chunks from {len(doc_files_info)} file(s).",
            "duration_ms": duration,
            "chunks_count": len(all_chunks),
            "docs_count": len(doc_files_info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
def search_docs(req: QueryRequest):
    if not all_chunks or retriever is None or generator is None:
        raise HTTPException(status_code=400, detail="No documents indexed in data directory.")
    
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        # Step 1: Hybrid Retrieval + Reranking with timing
        t_start = time.time()
        retrieved_docs = retriever.retrieve_and_rerank(query, initial_k=req.initial_k, final_k=req.final_k)
        t_retrieval = time.time()
        
        # Step 2: Grounded Synthesis with timing
        response = generator.answer_query(query, retrieved_docs)
        t_end = time.time()
        
        retrieval_ms = round((t_retrieval - t_start) * 1000)
        generation_ms = round((t_end - t_retrieval) * 1000)
        total_ms = round((t_end - t_start) * 1000)
        
        # Step 3: Format Context for UI Inspection Panel
        formatted_chunks = []
        for rank, doc in enumerate(retrieved_docs, start=1):
            chunk = doc['chunk']
            formatted_chunks.append({
                "rank": rank,
                "source": chunk.metadata.get('source', 'Unknown Document'),
                "section": chunk.metadata.get('Header 1', chunk.metadata.get('Header 2', 'General Section')),
                "rrf_score": round(doc.get('rrf_score', 0.0), 5),
                "cross_encoder_score": round(doc.get('cross_encoder_score', 0.0), 4),
                "content": chunk.page_content
            })
            
        return {
            "query": query,
            "answer": response["answer"],
            "sources": response["sources"],
            "retrieved_context": formatted_chunks,
            "metrics": {
                "retrieval_latency_ms": retrieval_ms,
                "generation_latency_ms": generation_ms,
                "total_latency_ms": total_ms,
                "chunks_evaluated": len(formatted_chunks)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search pipeline error: {str(e)}")

# Mount static folder for client-side assets (CSS, JS, images, icons)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")