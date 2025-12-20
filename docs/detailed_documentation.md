# 📚 API Documentation

Detailed technical documentation for the Multimodal RAG Chatbot.

---

## Table of Contents

1. [Module Overview](#module-overview)
2. [Core Module](#core-module)
   - [document_processor.py](#document_processorpy)
   - [embedding_engine.py](#embedding_enginepy)
   - [retrieval_engine.py](#retrieval_enginepy)
3. [Services Module](#services-module)
   - [llm_service.py](#llm_servicepy)
   - [rag_pipeline.py](#rag_pipelinepy)
4. [Components Module](#components-module)
5. [Configuration](#configuration)
6. [Data Flow Examples](#data-flow-examples)

---

## Module Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        app.py                                │
│                   (Streamlit Entry Point)                    │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
      │ components/ │ │  services/  │ │   config    │
      │  (UI Layer) │ │(Logic Layer)│ │ (Settings)  │
      └─────────────┘ └─────────────┘ └─────────────┘
                              │
                              ▼
                      ┌─────────────┐
                      │    core/    │
                      │(Processing) │
                      └─────────────┘
```

---

## Core Module

### document_processor.py

Handles document loading, parsing, and chunking.

#### Data Classes

##### `Chunk`

The fundamental unit of indexed content.

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Chunk:
    content: str                              # Text content or OCR text
    chunk_type: str                           # "text" | "table" | "image"
    page_number: int                          # Source page (1-indexed)
    source_file: str                          # Source filename
    metadata: dict = field(default_factory=dict)  # Additional metadata
    image_data: Optional[str] = None          # Base64-encoded image data
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | The text content. For images, contains OCR text prefixed with `[Image Content]` |
| `chunk_type` | `str` | One of `"text"`, `"table"`, or `"image"` |
| `page_number` | `int` | Page number in source document (1-indexed) |
| `source_file` | `str` | Filename of the source document |
| `metadata` | `dict` | Additional info like `image_index`, `width`, `height` |
| `image_data` | `str \| None` | Base64-encoded JPEG for image chunks, `None` for text/table |

**Example:**
```python
# Text chunk
text_chunk = Chunk(
    content="Solar panels convert sunlight into electricity...",
    chunk_type="text",
    page_number=3,
    source_file="solar_guide.pdf",
    metadata={},
    image_data=None
)

# Image chunk
image_chunk = Chunk(
    content="[Image Content]\nEfficiency chart showing...",
    chunk_type="image",
    page_number=5,
    source_file="solar_guide.pdf",
    metadata={"image_index": 0, "width": 800, "height": 600},
    image_data="iVBORw0KGgoAAAANSUhEUgA..."  # Base64
)
```

---

#### Functions

##### `load_document(file_path: str) -> List[Chunk]`

Main entry point for document processing.

```python
def load_document(file_path: str) -> List[Chunk]:
    """
    Load and process a document into chunks.
    
    Args:
        file_path: Absolute path to the document
        
    Returns:
        List of Chunk objects
        
    Raises:
        ValueError: If file type is not supported
        
    Supported formats:
        - PDF (.pdf)
        - Images (.png, .jpg, .jpeg)
        - Text (.txt)
    """
```

**Example:**
```python
from core.document_processor import load_document

chunks = load_document("/path/to/document.pdf")
print(f"Created {len(chunks)} chunks")

for chunk in chunks:
    print(f"  [{chunk.chunk_type}] Page {chunk.page_number}: {chunk.content[:50]}...")
```

---

##### `process_pdf(pdf_path: str) -> List[Chunk]`

Process a PDF file extracting text, images, and tables.

```python
def process_pdf(pdf_path: str) -> List[Chunk]:
    """
    Extract text, images, and tables from PDF using PyMuPDF.
    
    Processing steps:
    1. For each page:
       - Extract text → chunk into 1000-char segments
       - Extract images → OCR + base64 encode
       - Extract tables → convert to markdown
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of Chunk objects (mixed types)
    """
```

---

##### `process_image(image_path: str) -> List[Chunk]`

Process a standalone image file.

```python
def process_image(image_path: str) -> List[Chunk]:
    """
    Process an image file with OCR and CLIP embedding preparation.
    
    Args:
        image_path: Path to image file (PNG, JPG, JPEG)
        
    Returns:
        List containing one Chunk with image_data
    """
```

---

##### `run_ocr(image: PIL.Image) -> str`

Run OCR on an image.

```python
def run_ocr(image: Image.Image) -> str:
    """
    Extract text from image using EasyOCR.
    
    Args:
        image: PIL Image object
        
    Returns:
        Extracted text as string (empty string on failure)
    """
```

---

##### `chunk_text(text: str, page_num: int, source: str) -> List[Chunk]`

Split text into chunks using RecursiveCharacterTextSplitter.

```python
def chunk_text(text: str, page_num: int, source: str) -> List[Chunk]:
    """
    Split text into chunks with overlap.
    
    Parameters:
        - chunk_size: 1000 characters
        - chunk_overlap: 200 characters
        - separators: ["\\n\\n", "\\n", ". ", " "]
    
    Args:
        text: Text to chunk
        page_num: Page number for metadata
        source: Source filename for metadata
        
    Returns:
        List of text Chunk objects
    """
```

---

##### `table_to_markdown(table) -> str`

Convert PyMuPDF table to markdown format.

```python
def table_to_markdown(table) -> str:
    """
    Convert a PyMuPDF table object to markdown table format.
    
    Args:
        table: PyMuPDF table object
        
    Returns:
        Markdown-formatted table string
        
    Example output:
        | Header 1 | Header 2 |
        | --- | --- |
        | Cell 1 | Cell 2 |
    """
```

---

### embedding_engine.py

Handles CLIP embeddings and FAISS vector store.

#### Helper Functions

##### `chunk_to_dict(chunk: Chunk) -> dict`

Serialize Chunk to dictionary for pickling.

```python
def chunk_to_dict(chunk: Chunk) -> dict:
    """
    Convert Chunk to dictionary for safe serialization.
    
    This avoids dataclass pickling issues during hot-reload.
    """
```

##### `dict_to_chunk(d: dict) -> Chunk`

Deserialize dictionary back to Chunk.

```python
def dict_to_chunk(d: dict) -> Chunk:
    """
    Convert dictionary back to Chunk object.
    """
```

---

#### Classes

##### `EmbeddingEngine`

Manages CLIP model for text and image embeddings.

```python
class EmbeddingEngine:
    """
    CLIP embedding engine for multimodal content.
    
    Uses sentence-transformers/clip-ViT-B-32 model.
    Lazy-loads model on first use to avoid startup delays.
    
    Attributes:
        model: SentenceTransformer CLIP model (lazy-loaded)
        dimension: Embedding dimension (512)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        """
        Initialize embedding engine.
        
        Args:
            model_name: HuggingFace model identifier
        """
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text using CLIP text encoder.
        
        Args:
            text: Text to embed
            
        Returns:
            512-dimensional numpy array
        """
    
    def embed_image(self, image: PIL.Image) -> np.ndarray:
        """
        Embed image using CLIP image encoder.
        
        Args:
            image: PIL Image object
            
        Returns:
            512-dimensional numpy array
        """
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            (N, 512) numpy array
        """
    
    def embed_images_batch(self, images: List[PIL.Image]) -> np.ndarray:
        """
        Embed multiple images efficiently.
        
        Args:
            images: List of PIL Images
            
        Returns:
            (N, 512) numpy array
        """
```

**Usage Example:**
```python
from core.embedding_engine import EmbeddingEngine
from PIL import Image

engine = EmbeddingEngine()

# Embed text
text_vec = engine.embed_text("A photo of a cat")
print(text_vec.shape)  # (512,)

# Embed image
img = Image.open("cat.jpg")
img_vec = engine.embed_image(img)
print(img_vec.shape)  # (512,)

# Check similarity (should be high if image matches text)
similarity = np.dot(text_vec, img_vec)
print(f"Similarity: {similarity:.4f}")
```

---

##### `FAISSIndex`

Manages the FAISS vector store.

```python
class FAISSIndex:
    """
    FAISS index for multimodal chunk retrieval.
    
    Features:
        - Automatic text vs image detection
        - Images embedded using CLIP image encoder
        - Text embedded using CLIP text encoder
        - Cosine similarity search
        - Persistence to disk
    
    Attributes:
        dimension: Vector dimension (512)
        index: FAISS IndexFlatIP instance
        chunks: List of stored Chunk objects
        embedder: EmbeddingEngine instance
    """
    
    def __init__(self, dimension: int = 512):
        """Initialize empty FAISS index."""
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the index.
        
        Automatically detects chunk type:
        - Image chunks with image_data → CLIP image encoder
        - Text/table chunks → CLIP text encoder
        
        Args:
            chunks: List of Chunk objects to index
        """
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query: Text query (embedded with CLIP text encoder)
            k: Number of results to return
            
        Returns:
            List of (Chunk, similarity_score) tuples, sorted by score
        """
    
    def save(self, path: str) -> None:
        """
        Persist index to disk.
        
        Creates:
            - {path}/faiss.index: FAISS index file
            - {path}/chunks.pkl: Serialized chunks
        """
    
    def load(self, path: str) -> bool:
        """
        Load index from disk.
        
        Returns:
            True if successful, False if files missing/corrupted
        """
    
    def clear(self) -> None:
        """Clear all indexed data."""
```

**Usage Example:**
```python
from core.embedding_engine import FAISSIndex
from core.document_processor import load_document

# Create index
index = FAISSIndex()

# Add documents
chunks = load_document("document.pdf")
index.add_chunks(chunks)

# Search
results = index.search("What is machine learning?", k=5)
for chunk, score in results:
    print(f"[{score:.3f}] {chunk.content[:100]}...")

# Persist
index.save("data/faiss_index")

# Load later
new_index = FAISSIndex()
if new_index.load("data/faiss_index"):
    print("Index loaded successfully")
```

---

### retrieval_engine.py

Handles BM25, hybrid retrieval, and reranking.

#### Classes

##### `BM25Index`

Keyword-based search using BM25 algorithm.

```python
class BM25Index:
    """
    BM25 keyword search index.
    
    Complements semantic search with lexical matching.
    Useful for exact term matches and acronyms.
    """
    
    def build(self, chunks: List[Chunk]) -> None:
        """
        Build BM25 index from chunks.
        
        Tokenization: lowercase + whitespace split
        """
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Search using BM25 scoring.
        
        Returns:
            List of (Chunk, BM25_score) tuples
        """
    
    def save(self, path: str) -> None:
        """Save index to disk."""
    
    def load(self, path: str) -> bool:
        """Load index from disk."""
```

---

##### `HybridRetriever`

Combines FAISS and BM25 search results.

```python
class HybridRetriever:
    """
    Hybrid retrieval combining semantic and keyword search.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    
    Formula:
        RRF_score = sum(1 / (k + rank_i)) for each retriever
    
    Attributes:
        faiss: FAISSIndex instance
        bm25: BM25Index instance
    """
    
    def search(self, query: str, k: int = 10, alpha: float = 0.6) -> List[Chunk]:
        """
        Hybrid search with configurable weights.
        
        Args:
            query: Search query
            k: Number of results
            alpha: Weight for semantic search (0-1)
                   - 0.6 = 60% semantic, 40% keyword
                   
        Returns:
            List of Chunk objects (no scores, already ranked)
        """
```

**Example:**
```python
from core.embedding_engine import FAISSIndex
from core.retrieval_engine import BM25Index, HybridRetriever

faiss_idx = FAISSIndex()
bm25_idx = BM25Index()

# Build both indices
faiss_idx.add_chunks(chunks)
bm25_idx.build(chunks)

# Hybrid search
retriever = HybridRetriever(faiss_idx, bm25_idx)
results = retriever.search("machine learning algorithms", k=10, alpha=0.7)
```

---

##### `Reranker`

Cross-encoder reranking for precision.

```python
class Reranker:
    """
    Cross-encoder reranking using ms-marco-MiniLM-L-12-v2.
    
    More accurate than bi-encoder but slower.
    Used to re-score top candidates.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """Initialize with cross-encoder model (lazy-loaded)."""
    
    def rerank(self, query: str, chunks: List[Chunk], k: int = 3) -> List[Chunk]:
        """
        Rerank chunks using cross-encoder.
        
        Args:
            query: Original query
            chunks: Candidate chunks (e.g., top 10 from hybrid search)
            k: Number of top results to return
            
        Returns:
            Top k chunks, reordered by cross-encoder score
        """
```

---

##### `Compressor`

Extract relevant snippets from chunks.

```python
class Compressor:
    """
    Extract most relevant parts of chunks.
    
    Reduces context size while preserving relevant information.
    """
    
    def compress(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Compress chunks by extracting relevant sentences.
        
        Method: Keep sentences that share terms with query.
        Fallback: First N characters if no term overlap.
        """
```

---

## Services Module

### llm_service.py

Handles LLM integration, prompts, and conversation memory.

#### Classes

##### `Message`

Single conversation message.

```python
@dataclass
class Message:
    role: str      # "user" or "assistant"
    content: str   # Message content
```

##### `ConversationMemory`

Manages conversation history.

```python
class ConversationMemory:
    """
    Conversation memory for context-aware queries.
    
    Features:
        - Sliding window of last N turns
        - Formatted output for reformulation
        - Formatted output for context injection
    """
    
    def __init__(self, max_turns: int = 5):
        """
        Initialize memory.
        
        Args:
            max_turns: Max conversation turns to keep (1 turn = user + assistant)
        """
    
    def add_user_message(self, content: str) -> None:
        """Add user message to history."""
    
    def add_assistant_message(self, content: str) -> None:
        """Add assistant message to history."""
    
    def get_history_for_reformulation(self) -> str:
        """
        Get formatted history for query reformulation.
        
        Returns last 3 turns as "Human: ...\nAssistant: ..." format.
        """
    
    def get_history_for_context(self) -> List[Tuple[str, str]]:
        """
        Get history for prompt context.
        
        Returns last 2 turns as list of (role, content) tuples.
        """
    
    def clear(self) -> None:
        """Clear all history."""
```

---

#### Functions

##### `reformulate_query(query: str, memory: ConversationMemory) -> str`

Make follow-up questions standalone.

```python
def reformulate_query(query: str, memory: ConversationMemory) -> str:
    """
    Reformulate query using conversation history.
    
    Examples:
        "Tell me more about that" 
        → "Tell me more about [topic from previous answer]"
        
        "What about the second point?"
        → "What about [second point from previous context]?"
    
    Args:
        query: User's current query
        memory: Conversation history
        
    Returns:
        Standalone reformulated query
    """
```

---

##### `build_prompt(query, chunks, memory) -> Tuple[str, str]`

Construct RAG prompt.

```python
def build_prompt(
    query: str, 
    chunks: List[Chunk], 
    memory: Optional[ConversationMemory] = None
) -> Tuple[str, str]:
    """
    Build RAG prompt with context and instructions.
    
    Returns:
        Tuple of (system_prompt, user_message)
    
    System prompt includes:
        - Instructions for document-based answering
        - Citation format instructions
        - Retrieved context with source markers
        - Recent conversation history (if available)
    """
```

---

##### `generate_response(query, chunks, memory) -> Generator[str, None, None]`

Generate streaming LLM response.

```python
def generate_response(
    query: str, 
    chunks: List[Chunk], 
    memory: Optional[ConversationMemory] = None
) -> Generator[str, None, None]:
    """
    Generate streaming response from Gemini.
    
    Args:
        query: User question
        chunks: Retrieved context chunks
        memory: Conversation history
        
    Yields:
        Response tokens one at a time
    """
```

---

### rag_pipeline.py

Main RAG orchestration.

##### `RAGPipeline`

```python
class RAGPipeline:
    """
    Main RAG pipeline orchestrating all components.
    
    Responsibilities:
        1. Document processing and indexing
        2. Query reformulation
        3. Hybrid retrieval
        4. Reranking
        5. LLM response generation
        6. Memory management
    
    Attributes:
        faiss_index: FAISSIndex for vector search
        bm25_index: BM25Index for keyword search
        hybrid_retriever: HybridRetriever
        reranker: Reranker
        memory: ConversationMemory
        is_ready: Whether documents are loaded
    """
    
    def __init__(self):
        """Initialize pipeline and try to load existing indices."""
    
    def process_documents(self, file_paths: List[str]) -> int:
        """
        Process and index documents.
        
        Steps:
            1. Load each document
            2. Clear existing indices
            3. Add chunks to FAISS (with CLIP embeddings)
            4. Build BM25 index
            5. Create hybrid retriever
            6. Save indices to disk
        
        Args:
            file_paths: List of document paths
            
        Returns:
            Number of chunks created
        """
    
    def query(self, question: str) -> Generator[str, None, None]:
        """
        Run full RAG pipeline.
        
        Steps:
            1. Reformulate query using memory
            2. Hybrid search (FAISS + BM25)
            3. Rerank top candidates
            4. Generate LLM response
            5. Update memory
        
        Args:
            question: User's question
            
        Yields:
            Response tokens
        """
    
    def get_sources(self) -> List[Chunk]:
        """Get source chunks from last query."""
    
    def get_citations_markdown(self) -> str:
        """Get formatted citations."""
    
    def clear_memory(self) -> None:
        """Clear conversation memory only."""
    
    def clear(self) -> None:
        """Clear everything and reset."""
```

---

## Components Module

### sidebar.py

Streamlit sidebar with file upload and controls.

```python
def render_sidebar():
    """
    Render sidebar with:
        - File uploader (PDF, PNG, JPG, TXT)
        - Process Documents button
        - Memory turn count display
        - New Chat button (clears memory)
        - Clear All button (resets everything)
    """
```

### chat_interface.py

Chat message display.

```python
def render_chat():
    """
    Render chat interface with:
        - Message history
        - Chat input
        - Streaming response display
        - Source citations (expandable)
        - Query reformulation info (if different)
    """
```

### document_viewer.py

Source citation display.

```python
def display_sources(chunks: List[Chunk]):
    """
    Display source chunks in expandable sections.
    
    Shows:
        - Source file name
        - Page number
        - Content excerpt
        - Chunk type indicator
    """
```

---

## Configuration

### config.py

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # LLM Configuration
    google_api_key: str                    # Required: Gemini API key
    default_model: str = "gemini-2.0-flash"  # LLM model
    
    # Retrieval Configuration
    top_k_retrieval: int = 10              # Hybrid search candidates
    top_k_rerank: int = 3                  # Final results after reranking
    
    # Storage Paths
    upload_dir: str = "data/uploads"
    faiss_index_path: str = "data/faiss_index"
    bm25_index_path: str = "data/bm25_index"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Data Flow Examples

### Example 1: Indexing a PDF

```python
from core.document_processor import load_document
from core.embedding_engine import FAISSIndex
from core.retrieval_engine import BM25Index

# 1. Load and chunk document
chunks = load_document("research_paper.pdf")
# Result: ~50 chunks (text, table, image)

# 2. Create FAISS index with CLIP embeddings
faiss_idx = FAISSIndex()
faiss_idx.add_chunks(chunks)
# Images → CLIP image encoder
# Text → CLIP text encoder
# All in same 512-dim space

# 3. Create BM25 index
bm25_idx = BM25Index()
bm25_idx.build(chunks)
# Tokenized keyword index

# 4. Save both
faiss_idx.save("data/faiss_index")
bm25_idx.save("data/bm25_index")
```

### Example 2: Answering a Question

```python
from services.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# 1. User asks
question = "What does Figure 3 show?"

# 2. Pipeline processes
for token in pipeline.query(question):
    print(token, end="")
    
# Behind the scenes:
# - Query reformulated if needed
# - Hybrid search finds relevant chunks (including image chunks!)
# - Reranker selects top 3
# - Gemini generates answer with citations

# 3. Get sources
sources = pipeline.get_sources()
for src in sources:
    print(f"{src.source_file} p.{src.page_number}: {src.chunk_type}")
```

### Example 3: Follow-up Question

```python
# Previous: "What are the main findings?"
# Memory contains the Q&A

# User asks follow-up
question = "Can you explain the third one in more detail?"

# Reformulation happens:
# "Can you explain the third one..." 
# → "Can you explain the third main finding about [X] in more detail?"

for token in pipeline.query(question):
    print(token, end="")
```

---

## Appendix: Embedding Dimension Reference

| Model | Dimension | Purpose |
|-------|-----------|---------|
| CLIP ViT-B-32 | 512 | Text + Image embeddings |
| Cross-Encoder | N/A | Scores (query, doc) pairs |
| BM25 | N/A | Keyword matching scores |

---

*Last updated: December 2024*
