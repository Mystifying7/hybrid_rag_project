# 🔍 Smart Documentation Search Engine (Hybrid RAG)

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)
![AI](https://img.shields.io/badge/AI-Groq%20%7C%20Llama%203-orange.svg)

An enterprise-grade **Hybrid Retrieval-Augmented Generation (RAG)** pipeline designed to accurately search, retrieve, and synthesize technical documentation. Built as a B.Tech Minor Project, this system guarantees zero-hallucination answers backed by precise source citations.

---

## 🚀 The Problem & Solution
Standard search engines rely purely on exact keywords (missing contextual meaning), while standard Vector/Dense RAG pipelines often lose specific technical identifiers (like error codes or acronyms) in the embedding space.

**The Solution:** This project implements a **two-stage Retrieve-and-Rerank architecture**:
1. **Dual Retrieval:** Queries both a sparse lexical index (BM25) and a dense vector index (FAISS) simultaneously.
2. **Mathematical Fusion:** Merges the incompatible score scales using Reciprocal Rank Fusion (RRF).
3. **Neural Reranking:** Passes the fused candidates to a Cross-Encoder Transformer for deep token-level self-attention.
4. **Grounded Synthesis:** Feeds the mathematically optimal context to an LLM with strict temperature controls to generate a deterministic, cited answer.

---

## ✨ Key Features
* **Markdown-Aware Chunking:** Preserves document hierarchy and section headers as metadata instead of blindly slicing text.
* **BM25 Sparse Index:** Optimizes for high-frequency keyword precision and exact code matching.
* **FAISS Dense Index (`all-MiniLM-L6-v2`):** Enables sub-second semantic similarity search.
* **Reciprocal Rank Fusion (RRF):** Non-parametric rank merging for hybrid search.
* **Cross-Encoder Reranker (`ms-marco`):** Computes deep semantic relevance across query-document pairs.
* **Glass-Box Web UI:** Custom FastAPI and Vanilla JS frontend that visualizes Cross-Encoder logits, RRF scores, and retrieved markdown context in real-time.

---

## 🧠 System Architecture

```text
[ Raw Technical Docs (.md) ]
             │
             ▼
[ MarkdownHeaderTextSplitter + Recursive Splitter ]
             │
     ┌───────┴──────────────────┐
     ▼                          ▼
[ FAISS Vector Store ]   [ BM25 Sparse Index ]
(Dense Embeddings)       (Lexical TF-IDF)
     │                          │
     └───────┬──────────────────┘
             ▼
[ Reciprocal Rank Fusion (RRF: k=60) ]
             │
             ▼ 
[ Cross-Encoder Reranker ] (Self-Attention Scoring)
             │
             ▼ (Top 3 Chunks)
[ Groq LLM (Llama 3) with Strict Guardrail Prompt ]
             │
             ▼
[ Cited, Hallucination-Free Answer + UI Inspector ]
```
---

## 🛠️ Tech Stack
-> Backend: Python, FastAPI, Uvicorn

-> AI & NLP: LangChain, SentenceTransformers, FAISS, Rank_BM25, HuggingFace

-> LLM Inference: Groq Cloud API (Llama 3 / 3.3)

-> Frontend: HTML5, CSS3, Vanilla JavaScript

---

## ⚙️ Quick Start (Local Setup)
1. Clone the Repository
```text
git clone https://github.com/Mystifying7/hybrid_rag_project.git
(https://github.com/Mystifying7/hybrid_rag_project.git)
cd hybrid_rag_project
```
2. Set Up Virtual Environment
```text
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```
3. Install Dependencies
```text
pip install -r requirements.txt
```
4. Configure API Keys \
Create a .env file in the root directory and add your Groq API key:
```text
GROQ_API_KEY=your_api_key_here
```
5. Add Sample Data \
Place your Markdown (.md) documentation files inside the data/sample_docs/ directory. (Tip: Clone open-source repos like FastAPI's docs to test at scale!)

6. Run the Server
```text
python server.py
```
Open your browser and go to : 
```text
http://localhost:8000 
```
---
## 📂 Project Structure
```text
hybrid_rag_project/
├── data/
│   └── sample_docs/         # Place markdown files here
├── src/
│   ├── ingestion.py         # Structure-aware document chunking
│   ├── indexer.py           # FAISS and BM25 index builders
│   ├── retriever.py         # RRF and Cross-Encoder logic
│   └── generator.py         # LLM prompt synthesis & guardrails
├── static/
│   ├── index.html           # Product Landing Page
|   ├── app.js 
│   └── style.css            # Project styling
├── server.py                # FastAPI routing and application state
├── .env                     # Add you api key of grok here
├── app.py
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```
