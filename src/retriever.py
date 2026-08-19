# src/retriever.py
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class HybridRetriever:
    def __init__(self, indexer, cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the retriever with our pre-built DualIndexer and a Cross-Encoder.
        """
        self.indexer = indexer
        print(f"Loading Cross-Encoder model: {cross_encoder_model}...")
        self.reranker = CrossEncoder(cross_encoder_model)

    def reciprocal_rank_fusion(self, faiss_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
        """
        Merges FAISS and BM25 results using RRF.
        """
        fusion_scores = {}
        chunk_map = {}
        
        # Helper function to process results
        def process_results(results):
            for res in results:
                # Using page_content as a unique identifier for simplicity.
                # In production, use a UUID attached to the chunk metadata.
                content = res['chunk'].page_content
                rank = res['rank']
                
                if content not in fusion_scores:
                    fusion_scores[content] = 0.0
                    chunk_map[content] = res['chunk']
                
                # Apply RRF Formula
                fusion_scores[content] += 1.0 / (k + rank)

        # Process both result sets
        process_results(faiss_results)
        process_results(bm25_results)
        
        # Sort by the new RRF fusion score descending
        fused_results = [
            {"chunk": chunk_map[content], "rrf_score": score}
            for content, score in fusion_scores.items()
        ]
        fused_results = sorted(fused_results, key=lambda x: x["rrf_score"], reverse=True)
        return fused_results

    def retrieve_and_rerank(self, query: str, initial_k: int = 10, final_k: int = 3) -> List[Dict]:
        """
        The master pipeline:
        1. Retrieve top `initial_k` from FAISS and BM25.
        2. Fuse them using RRF.
        3. Rerank the fused results using the Cross-Encoder.
        4. Return the top `final_k` results.
        """
        # Step 1: Base Retrieval
        faiss_res = self.indexer.search_faiss(query, top_k=initial_k)
        bm25_res = self.indexer.search_bm25(query, top_k=initial_k)
        
        # Step 2: RRF Fusion
        fused_res = self.reciprocal_rank_fusion(faiss_res, bm25_res)
        
        # If we have no results, return empty
        if not fused_res:
            return []
            
        # Step 3: Cross-Encoder Reranking
        # Prepare the pairs for the cross-encoder: [[query, doc1], [query, doc2], ...]
        cross_inp = [[query, item["chunk"].page_content] for item in fused_res]
        
        # Predict relevance scores
        cross_scores = self.reranker.predict(cross_inp)
        
        # Attach new scores and sort
        for i, score in enumerate(cross_scores):
            fused_res[i]["cross_encoder_score"] = float(score)
            
        # Sort by cross-encoder score descending
        reranked_results = sorted(fused_res, key=lambda x: x["cross_encoder_score"], reverse=True)
        
        # Step 4: Return Top K
        return reranked_results[:final_k]

if __name__ == "__main__":
    # --- Local Testing Block ---
    import os
    from ingestion import process_markdown_file
    from indexer import DualIndexer
    
    test_file = "../data/sample_docs/test_doc.md"
    
    print("Setting up pipeline (this might take a moment to load models)...\n")
    chunks = process_markdown_file(test_file, chunk_size=150, chunk_overlap=20)
    
    indexer = DualIndexer()
    indexer.fit(chunks)
    
    retriever = HybridRetriever(indexer=indexer)
    
    query = "How do I upgrade the database?"
    print(f"\n--- Testing Pipeline Query: '{query}' ---")
    
    final_results = retriever.retrieve_and_rerank(query, initial_k=5, final_k=2)
    
    for i, res in enumerate(final_results):
        print(f"\n--- Final Rank {i+1} ---")
        print(f"RRF Score: {res['rrf_score']:.4f}")
        print(f"Cross-Encoder Score: {res['cross_encoder_score']:.4f}")
        print(f"Content: {res['chunk'].page_content}")