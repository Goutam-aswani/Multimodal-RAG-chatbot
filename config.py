from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    google_api_key: str = ""
    groq_api_key: str = ""
    default_model: str = "gemini-2.5-flash"
    embedding_model: str = "sentence-transformers/clip-ViT-B-32"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 10
    top_k_rerank: int = 3
    similarity_threshold: float = 0.7
    upload_dir: str = "data/uploads"
    faiss_index_path: str = "data/faiss_index"
    bm25_index_path: str = "data/bm25_index"
    class Config:
        env_file = ".env"

settings = Settings()
