# app.py
import os
import streamlit as st
from dotenv import load_dotenv

# Import our custom modular pipeline
from src.ingestion import process_markdown_file
from src.indexer import DualIndexer
from src.retriever import HybridRetriever
from src.generator import RAGGenerator

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Smart Documentation Search Engine",
    page_icon="🔍",
    layout="wide"
)

# ---------------------------------------------------------
# Resource Caching: Load Models Once into RAM
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_pipeline(docs_path: str):
    """
    Ingests docs, builds FAISS + BM25 indices, and caches the retriever & generator in memory.
    """
    # 1. Ingestion
    all_chunks = []
    if os.path.exists(docs_path):
        for root, _, files in os.walk(docs_path):
            for file in files:
                if file.endswith(".md") or file.endswith(".txt"):
                    full_path = os.path.join(root, file)
                    chunks = process_markdown_file(full_path, chunk_size=300, chunk_overlap=40)
                    all_chunks.extend(chunks)
    
    if not all_chunks:
        return None, None, 0

    # 2. Dual Indexing
    indexer = DualIndexer(embedding_model_name='all-MiniLM-L6-v2')
    indexer.fit(all_chunks)

    # 3. Retriever
    retriever = HybridRetriever(indexer=indexer)

    # 4. Generator
    # Use the same model string that verified successfully in Milestone 5
    generator = RAGGenerator(model_name="openai/gpt-oss-120b", temperature=0.0)

    return retriever, generator, len(all_chunks)


# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
st.title("🔍 Smart Documentation Search Engine")
st.caption("Hybrid RAG Pipeline: BM25 (Lexical) + FAISS (Dense) + Cross-Encoder Reranker + LLM Guardrails")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration & Index")
    docs_dir = st.text_input("Documentation Directory", value="data/sample_docs")
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        initial_k = st.slider("Initial K (Fusion)", min_value=2, max_value=15, value=6)
    with col_k2:
        final_k = st.slider("Final K (Reranked)", min_value=1, max_value=5, value=2)
    
    reindex_btn = st.button("🔄 Rebuild Search Index", use_container_width=True)
    if reindex_btn:
        st.cache_resource.clear()
        st.rerun()

# Pipeline Initialization
with st.spinner("Initializing indexing pipeline and loading models..."):
    retriever, generator, total_chunks = initialize_pipeline(docs_dir)

if total_chunks == 0:
    st.warning(f"No documents found in `{docs_dir}`. Please add `.md` files to the directory.")
    st.stop()

st.sidebar.success(f"Indexed **{total_chunks}** chunks ready for hybrid retrieval.")

# Main Search Bar
user_query = st.text_input("Enter your technical question / error code / query:", placeholder="e.g. How do I upgrade the database schema?")

if user_query:
    col_left, col_right = st.columns([3, 2])
    
    with st.spinner("Searching, reranking, and generating grounded response..."):
        # 1. Retrieve & Rerank
        retrieved_docs = retriever.retrieve_and_rerank(user_query, initial_k=initial_k, final_k=final_k)
        
        # 2. Synthesize
        llm_response = generator.answer_query(user_query, retrieved_docs)

    # Left Column: Synthesized Answer
    with col_left:
        st.subheader("💡 Answer")
        st.markdown(llm_response["answer"])
        
        if llm_response["sources"]:
            st.divider()
            st.markdown("**Cited Sources:**")
            for src in llm_response["sources"]:
                st.caption(f"📌 `{src}`")

    # Right Column: Context Inspector (Explainability)
    with col_right:
        st.subheader("🔬 Retrieved & Reranked Context")
        st.caption(f"Top {len(retrieved_docs)} candidate chunks sent to LLM:")
        
        for rank, item in enumerate(retrieved_docs, start=1):
            chunk = item["chunk"]
            ce_score = item.get("cross_encoder_score", 0.0)
            rrf_score = item.get("rrf_score", 0.0)
            
            with st.expander(f"Rank {rank} | Relevance Score: {ce_score:.4f}", expanded=(rank == 1)):
                st.markdown(f"**Source File:** `{chunk.metadata.get('source', 'Unknown')}`")
                st.markdown(f"**Section:** `{chunk.metadata.get('Header 1', chunk.metadata.get('Header 2', 'N/A'))}`")
                st.markdown(f"**RRF Score:** `{rrf_score:.5f}` | **Cross-Encoder Logit:** `{ce_score:.4f}`")
                st.divider()
                st.code(chunk.page_content, language="markdown")