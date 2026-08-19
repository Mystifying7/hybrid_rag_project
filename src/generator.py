import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

class RAGGenerator:
    def __init__(self, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.0):
        """
        Initializes the LLM and the strict system prompt for grounding.
        """
        print(f"Initializing LLM: {model_name} with temperature {temperature}")
        # Make sure GROQ_API_KEY is in your .env file
        self.llm = ChatGroq(model=model_name, temperature=temperature)
        
        # The System Prompt acts as our Hallucination Guardrail
        self.system_prompt = """You are a highly precise technical assistant. 
        Your task is to answer the user's question based strictly on the provided context below.
        
        Rules:
        1. If the answer is not contained in the context, say exactly: "I'm sorry, I cannot find the answer in the provided documentation."
        2. Do NOT use your outside knowledge. Do NOT make up facts.
        3. ALWAYS cite the source file and section name at the end of your answer if you used it.
        
        Context Data:
        {context}
        """
        
        # Compile the template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}")
        ])
        
        # Create the LangChain pipeline
        self.chain = self.prompt_template | self.llm

    def _format_context(self, retrieved_docs: List[Dict]) -> Tuple[str, List[str]]:
        """
        Takes the reranked dictionary objects and formats them into a single string 
        for the LLM, preserving metadata for citations.
        """
        context_parts = []
        unique_sources = set()
        
        for doc in retrieved_docs:
            chunk = doc['chunk']
            meta = chunk.metadata
            
            # Extract metadata safely
            source_file = meta.get('source', 'Unknown Document')
            section = meta.get('Header 1', meta.get('Header 2', 'General Section'))
            
            # Format the text with its metadata tag
            formatted_chunk = f"--- [Source: {source_file} | Section: {section}] ---\n{chunk.page_content}"
            context_parts.append(formatted_chunk)
            unique_sources.add(f"{source_file} ({section})")
            
        return "\n\n".join(context_parts), list(unique_sources)

    def answer_query(self, query: str, retrieved_docs: List[Dict]) -> Dict:
        """
        The final pipeline step: feeds context and query to the LLM.
        """
        # 1. Format the retrieved docs
        formatted_context, sources = self._format_context(retrieved_docs)
        
        # 2. Invoke the LLM
        response = self.chain.invoke({
            "context": formatted_context,
            "question": query
        })
        
        return {
            "answer": response.content,
            "sources": sources
        }

if __name__ == "__main__":
    # --- Local Testing Block ---
    import os
    from ingestion import process_markdown_file
    from indexer import DualIndexer
    from retriever import HybridRetriever
    
    test_file = "../data/sample_docs/test_doc.md"
    
    print("\n--- Booting up Full RAG Pipeline ---")
    # 1. Ingest
    chunks = process_markdown_file(test_file, chunk_size=150, chunk_overlap=20)
    # 2. Index
    indexer = DualIndexer()
    indexer.fit(chunks)
    # 3. Retrieve
    retriever = HybridRetriever(indexer=indexer)
    # 4. Generate
    generator = RAGGenerator()
    
    # Let's ask a valid question
    query_1 = "What database are we using and how do I upgrade it?"
    print(f"\n[Query]: {query_1}")
    docs_1 = retriever.retrieve_and_rerank(query_1, initial_k=5, final_k=3)
    result_1 = generator.answer_query(query_1, docs_1)
    print(f"\n[LLM Answer]:\n{result_1['answer']}")
    
    # Let's ask an out-of-scope question to test our guardrails
    query_2 = "How do I configure the Redis cache?"
    print(f"\n[Query]: {query_2}")
    docs_2 = retriever.retrieve_and_rerank(query_2, initial_k=5, final_k=3)
    result_2 = generator.answer_query(query_2, docs_2)
    print(f"\n[LLM Answer]:\n{result_2['answer']}")