# src/indexer.py
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from langchain_core.documents import Document

class DualIndexer:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes our two search engines:
        1. SentenceTransformer for Dense Vectors
        2. BM25 for Sparse Lexical Search
        """
        print(f"Loading embedding model: {embedding_model_name}...")
        # all-MiniLM-L6-v2 is small, fast, and generates 384-dimensional vectors
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS Index (L2 distance is standard and fast)
        self.faiss_index = faiss.IndexFlatL2(self.vector_dimension)
        
        self.bm25_index = None
        self.documents: List[Document] = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25: lowercase and split by spaces."""
        return text.lower().split()

    def fit(self, chunks: List[Document]):
        """
        Takes a list of document chunks and indexes them in BOTH FAISS and BM25.
        """
        self.documents = chunks
        texts = [chunk.page_content for chunk in chunks]
        
        # --- 1. Build FAISS Dense Index ---
        print("Generating dense vectors...")
        # Convert text to vectors
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        # Add vectors to FAISS
        self.faiss_index.add(embeddings)
        print(f"Added {self.faiss_index.ntotal} vectors to FAISS.")

        # --- 2. Build BM25 Sparse Index ---
        print("Generating BM25 sparse index...")
        tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print("BM25 index built successfully.")

    def search_faiss(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches the dense FAISS index."""
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True)
        # distances and indices of the nearest neighbors
        distances, indices = self.faiss_index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # -1 means no result found
                results.append({
                    "chunk": self.documents[idx],
                    "score": float(distances[0][i]), # Lower distance is better for L2
                    "rank": i + 1
                })
        return results

    def search_bm25(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches the sparse BM25 index."""
        tokenized_query = self._tokenize(query)
        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get the indices of the top_k highest scores
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_n_indices):
            if scores[idx] > 0: # Only return if there is some matching score
                results.append({
                    "chunk": self.documents[idx],
                    "score": float(scores[idx]), # Higher score is better for BM25
                    "rank": i + 1
                })
        return results

if __name__ == "__main__":
    # --- Local Testing Block ---
    # We will import our chunker from Milestone 2
    import os
    from ingestion import process_markdown_file
    
    test_file = "../data/sample_docs/test_doc.md"
    if not os.path.exists(test_file):
        print("Please run Milestone 2 first to generate the test file.")
    else:
        chunks = process_markdown_file(test_file, chunk_size=150, chunk_overlap=20)
        
        # Initialize and fit our dual indexer
        indexer = DualIndexer()
        indexer.fit(chunks)
        
        # Test Query
        query = "How do I upgrade the database?"
        print(f"\n--- Testing Query: '{query}' ---")
        
        print("\nFAISS Results (Dense - Semantic):")
        faiss_results = indexer.search_faiss(query, top_k=2)
        for res in faiss_results:
            print(f"Rank {res['rank']} | L2 Distance: {res['score']:.4f}\n{res['chunk'].page_content}\n")
            
        print("\nBM25 Results (Sparse - Lexical):")
        bm25_results = indexer.search_bm25(query, top_k=2)
        for res in bm25_results:
            print(f"Rank {res['rank']} | BM25 Score: {res['score']:.4f}\n{res['chunk'].page_content}\n")