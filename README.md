# 🔍 Smart Documentation Search Engine

An advanced, production-grade Hybrid Retrieval-Augmented Generation (RAG) pipeline built to accurately search and synthesize technical documentation.

## 🚀 Features
* **Dual-Index Retrieval:** Combines sparse lexical search (BM25) for exact keyword matches and dense vector search (FAISS + `all-MiniLM-L6-v2`) for semantic understanding.
* **Mathematical Rank Fusion:** Uses Reciprocal Rank Fusion (RRF) to merge disparate scoring scales.
* **Cross-Encoder Reranking:** Employs `ms-marco-MiniLM-L-6-v2` to compute self-attention across query-document pairs for maximum context relevance.
* **Hallucination Guardrails:** Strict zero-temperature LLM generation using Llama 3 with forced source-citation constraints.
* **Decoupled Architecture:** Asynchronous FastAPI backend with a responsive HTML/JS vanilla frontend.

## 🧠 System Architecture
1. **Ingestion:** Markdown files are split using structure-aware chunking (preserving headers).
2. **Retrieval:** $O(1)$ BM25 lookup + $O(\log N)$ FAISS L2 distance search.
3. **Reranking:** Cross-encoder strictly reranks the top 10 candidates down to the top 3.
4. **Generation:** Grounded context is passed to the LLM to synthesize the final cited answer.

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn
* **AI/ML:** LangChain, SentenceTransformers, FAISS, Rank_BM25, Groq API (Llama 3)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript

## ⚙️ Quick Start (Local Setup)

1. **Clone and setup environment:**
   ```bash
   git clone https://github.com/Mystifying7/hybrid_rag_project.git
   cd hybrid_rag_project
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
