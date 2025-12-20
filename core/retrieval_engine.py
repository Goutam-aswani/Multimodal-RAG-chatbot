"""
Retrieval Engine Module
Classes:
- BM25Index: Keyword-based search
- HybridRetriever: Combine vector + BM25
- Reranker: Cross-encoder reranking
- Compressor: Extract relevant snippets

Functions:
- hybrid_search(query, k) → List[Chunk]
- rerank(query, chunks, k) → List[Chunk]
- compress(query, chunks) → List[Chunk]
"""

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List, Tuple
import numpy as np
import pickle
import os
from core.document_processor import Chunk
from core.embedding_engine import FAISSIndex, chunk_to_dict, dict_to_chunk

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunks: List[Chunk] = []
    
    def build(self, chunks: List[Chunk]):
        """Build BM25 index from chunks."""
        self.chunks = chunks
        tokenized = [c.content.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Search using BM25."""
        if self.bm25 is None or not self.chunks:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        k = min(k, len(self.chunks))
        
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]
    
    def save(self, path: str):
        """Save BM25 index to disk."""
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/bm25.pkl", "wb") as f:
            data = {
                'bm25': self.bm25,
                'chunks': [chunk_to_dict(c) for c in self.chunks]
            }
            pickle.dump(data, f)
    
    def load(self, path: str) -> bool:
        """Load BM25 index from disk. Returns True if successful."""
        bm25_path = f"{path}/bm25.pkl"
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, "rb") as f:
                    data = pickle.load(f)

                self.bm25 = data.get('bm25')
                chunks_as_dicts = data.get('chunks', [])
                self.chunks = [dict_to_chunk(d) for d in chunks_as_dicts]
                return True
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Failed to load BM25 index from {bm25_path}: {e}")
                return False
            except Exception as e:
                print(f"Unexpected error loading BM25 index from {bm25_path}: {e}")
                return False
        return False

class HybridRetriever:
    def __init__(self, faiss_index: FAISSIndex, bm25_index: BM25Index):
        self.faiss = faiss_index
        self.bm25 = bm25_index
    
    def search(self, query: str, k: int = 10, alpha: float = 0.6) -> List[Chunk]:
        """
        Hybrid search with RRF (Reciprocal Rank Fusion).
        alpha: Weight for vector search (1-alpha for BM25)
        """
        vector_results = self.faiss.search(query, k * 2)
        bm25_results = self.bm25.search(query, k * 2)
        
        rrf_scores = {}
        chunk_map = {}
        rrf_k = 60          
        for rank, (chunk, _) in enumerate(vector_results):
            chunk_id = id(chunk)
            chunk_map[chunk_id] = chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + alpha / (rrf_k + rank + 1)
        
        for rank, (chunk, _) in enumerate(bm25_results):
            chunk_id = id(chunk)
            chunk_map[chunk_id] = chunk
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (1 - alpha) / (rrf_k + rank + 1)
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        return [chunk_map[cid] for cid in sorted_ids[:k]]

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self._model = None
        self.model_name = model_name
    
    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model
    
    def rerank(self, query: str, chunks: List[Chunk], k: int = 3) -> List[Chunk]:
        """Rerank chunks using cross-encoder."""
        if not chunks:
            return []
        
        pairs = [(query, c.content) for c in chunks]
        scores = self.model.predict(pairs)
        
        k = min(k, len(chunks))
        
        ranked_indices = np.argsort(scores)[::-1][:k]
        return [chunks[i] for i in ranked_indices]

class Compressor:
    """Extract only the most relevant snippets from chunks."""
    
    def __init__(self, max_tokens: int = 500):
        self.max_tokens = max_tokens
    
    def compress(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Compress chunks to extract most relevant parts.
        Simple implementation - can be enhanced with LLM-based extraction.
        """
        compressed = []
        query_terms = set(query.lower().split())
        
        for chunk in chunks:
            sentences = chunk.content.split('. ')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_terms = set(sentence.lower().split())
                if query_terms & sentence_terms:
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                compressed_content = '. '.join(relevant_sentences)
            else:
                compressed_content = chunk.content[:self.max_tokens]
            
            compressed.append(Chunk(
                content=compressed_content,
                chunk_type=chunk.chunk_type,
                page_number=chunk.page_number,
                source_file=chunk.source_file,
                metadata=chunk.metadata
            ))
        return compressed
