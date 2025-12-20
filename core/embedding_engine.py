"""
Embedding Engine Module
Classes:
- EmbeddingEngine: Generate CLIP embeddings
- FAISSIndex: Manage vector store

Functions:
- embed_text(text) → np.array
- embed_image(image) → np.array
- add_chunks(chunks) → None
- search(query, k) → List[Chunk]
- save_index() → None
- load_index() → None
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import pickle
import os
import base64
import io
from PIL import Image as PILImage
from core.document_processor import Chunk


def chunk_to_dict(chunk: Chunk):
    return {
        'content': chunk.content,
        'chunk_type': chunk.chunk_type,
        'page_number': chunk.page_number,
        'source_file': chunk.source_file,
        'metadata': chunk.metadata,
        'image_data': chunk.image_data
    }


def dict_to_chunk(d):
    return Chunk(
        content=d.get('content', ''),
        chunk_type=d.get('chunk_type', 'text'),
        page_number=d.get('page_number', 1),
        source_file=d.get('source_file', ''),
        metadata=d.get('metadata', {}),
        image_data=d.get('image_data')
    )

class EmbeddingEngine:
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        self._model_name = model_name
        self.model = None
        self.dimension = 512
    
    def _ensure_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self._model_name)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using CLIP text encoder."""
        self._ensure_model()
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_image(self, image) -> np.ndarray:
        """
        Embed an image using CLIP image encoder.
        Args:
            image: PIL.Image.Image object
        Returns:
            np.ndarray: Image embedding vector
        """
        self._ensure_model()
        return self.model.encode(image, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently."""
        self._ensure_model()
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32)
    
    def embed_images_batch(self, images: list) -> np.ndarray:
        """
        Embed multiple images efficiently.
        Args:
            images: List of PIL.Image.Image objects
        Returns:
            np.ndarray: Array of image embeddings
        """
        self._ensure_model()
        return self.model.encode(images, convert_to_numpy=True, batch_size=16)

class FAISSIndex:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []
        self.embedder = EmbeddingEngine()
    
    def add_chunks(self, chunks: List[Chunk]):
        """
        Add chunks to the index.
        For image chunks with image_data, embed the actual image using CLIP.
        For text/table chunks, embed the text content.
        """
        if not chunks:
            return
        
        all_embeddings = []
        
        text_chunks = []
        image_chunks = []
        
        for chunk in chunks:
            if chunk.chunk_type == "image" and chunk.image_data:
                image_chunks.append(chunk)
            else:
                text_chunks.append(chunk)
        
        if text_chunks:
            texts = [c.content for c in text_chunks]
            text_embeddings = self.embedder.embed_batch(texts)
            for i, chunk in enumerate(text_chunks):
                all_embeddings.append((chunk, text_embeddings[i]))
        
        if image_chunks:
            pil_images = []
            for chunk in image_chunks:
                try:
                    image_bytes = base64.b64decode(chunk.image_data)
                    pil_image = PILImage.open(io.BytesIO(image_bytes))
                    pil_images.append(pil_image)
                except Exception as e:
                    text_emb = self.embedder.embed_text(chunk.content)
                    all_embeddings.append((chunk, text_emb))
                    continue
            
            if pil_images:
                # Embed all images in batch
                image_embeddings = self.embedder.embed_images_batch(pil_images)
                for i, chunk in enumerate(image_chunks[:len(pil_images)]):
                    all_embeddings.append((chunk, image_embeddings[i]))
        
        if all_embeddings:
            embeddings_array = np.array([emb for _, emb in all_embeddings]).astype('float32')
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            self.chunks.extend([chunk for chunk, _ in all_embeddings])
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks."""
        if self.index.ntotal == 0:
            return []
            
        query_embedding = self.embedder.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        k = min(k, self.index.ntotal)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, path: str):
        """Save index to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/faiss.index")
        chunks_as_dicts = [chunk_to_dict(c) for c in self.chunks]
        with open(f"{path}/chunks.pkl", "wb") as f:
            pickle.dump(chunks_as_dicts, f)
    
    def load(self, path: str) -> bool:
        """Load index from disk. Returns True if successful."""
        index_path = f"{path}/faiss.index"
        chunks_path = f"{path}/chunks.pkl"
        
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            try:
                with open(chunks_path, "rb") as f:
                    chunks_as_dicts = pickle.load(f)
                self.chunks = [dict_to_chunk(d) for d in chunks_as_dicts]
                return True
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Failed to load chunks from {chunks_path}: {e}")
                return False
            except Exception as e:
                print(f"Unexpected error loading chunks from {chunks_path}: {e}")
                return False
        return False
    
    def clear(self):
        """Clear the index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
