"""
RAG Pipeline Module
Class: RAGPipeline
- process_documents(files) → None
- query(question) → Generator[str]
- get_sources() → List[Chunk]
"""

from core.document_processor import load_document, Chunk
from core.embedding_engine import FAISSIndex
from core.retrieval_engine import BM25Index, HybridRetriever, Reranker
from services.llm_service import (generate_response, format_citations,ConversationMemory,reformulate_query)
from config import settings
from typing import List, Generator, Optional
import os

class RAGPipeline:
    def __init__(self):
        self.faiss_index = FAISSIndex()
        self.bm25_index = BM25Index()
        self.hybrid_retriever = None
        self.reranker = Reranker()
        self.last_sources: List[Chunk] = []
        self.is_ready = False
        self.memory = ConversationMemory(max_turns=5)
        self.last_reformulated_query: Optional[str] = None
        self._load_existing_indices()
    
    def _load_existing_indices(self):
        """Try to load existing indices from disk."""
        faiss_loaded = self.faiss_index.load(settings.faiss_index_path)
        bm25_loaded = self.bm25_index.load(settings.bm25_index_path)
        
        if faiss_loaded and bm25_loaded:
            self.hybrid_retriever = HybridRetriever(self.faiss_index, self.bm25_index)
            self.is_ready = True
    
    def process_documents(self, file_paths: List[str]) -> int:
        """Process and index documents."""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                chunks = load_document(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if not all_chunks:
            return 0
        self.faiss_index.clear()
        self.faiss_index.add_chunks(all_chunks)
        self.bm25_index.build(all_chunks)
        self.hybrid_retriever = HybridRetriever(self.faiss_index, self.bm25_index)
        
        os.makedirs(settings.faiss_index_path, exist_ok=True)
        os.makedirs(settings.bm25_index_path, exist_ok=True)
        
        self.faiss_index.save(settings.faiss_index_path)
        self.bm25_index.save(settings.bm25_index_path)
        
        self.is_ready = True
        return len(all_chunks)
    
    def query(self, question: str) -> Generator[str, None, None]:
        """Run full RAG pipeline with memory-aware query reformulation and stream response."""
        if not self.is_ready:
            yield "Please upload documents first."
            return
        
        try:
            reformulated_query = reformulate_query(question, self.memory)
            self.last_reformulated_query = reformulated_query
            
            self.memory.add_user_message(question)
            
            candidates = self.hybrid_retriever.search(reformulated_query, k=settings.top_k_retrieval)
            
            if not candidates:
                response = "No relevant information found in the documents."
                self.memory.add_assistant_message(response)
                yield response
                return
            
            top_chunks = self.reranker.rerank(reformulated_query, candidates, k=settings.top_k_rerank)
            self.last_sources = top_chunks
            
            full_response = ""
            for token in generate_response(question, top_chunks, self.memory):
                full_response += token
                yield token
            
            self.memory.add_assistant_message(full_response)
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            self.memory.add_assistant_message(error_msg)
            yield error_msg
    
    def get_sources(self) -> List[Chunk]:
        """Get sources from last query."""
        return self.last_sources
    
    def get_citations_markdown(self) -> str:
        """Get formatted citations."""
        return format_citations(self.last_sources)
    
    def get_last_reformulated_query(self) -> Optional[str]:
        """Get the last reformulated query (useful for debugging)."""
        return self.last_reformulated_query
    
    def clear_memory(self):
        """Clear conversation memory only."""
        self.memory.clear()
        self.last_reformulated_query = None
    
    def clear(self):
        """Clear all indices, memory, and reset state."""
        self.faiss_index.clear()
        self.bm25_index = BM25Index()
        self.hybrid_retriever = None
        self.last_sources = []
        self.is_ready = False
        self.memory.clear()
        self.last_reformulated_query = None
