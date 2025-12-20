# 🔍 Multimodal RAG Chatbot# Multimodal RAG Chatbot



A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that processes multimodal documents (PDFs with text, images, and tables) and answers questions using Google's Gemini LLM.A Streamlit-based multimodal RAG (Retrieval-Augmented Generation) chatbot that can process PDFs, images, and text files.



---## Features



## 📋 Table of Contents- **Document Processing**: Load PDFs, images, and text files with OCR support

- **Hybrid Search**: Combines FAISS vector search with BM25 keyword search

- [Features](#-features)- **Reranking**: Cross-encoder reranking for improved relevance

- [Architecture](#-architecture)- **Streaming Responses**: Real-time LLM response streaming

- [Technology Stack](#-technology-stack)- **Citation Tracking**: Source attribution for all responses

- [Installation](#-installation)

- [Configuration](#-configuration)## Setup

- [Usage](#-usage)

- [Project Structure](#-project-structure)1. Install dependencies:

- [How It Works](#-how-it-works)```bash

- [API Reference](#-api-reference)pip install -r requirements.txt

- [Troubleshooting](#-troubleshooting)```



---2. Configure environment variables in `.env`:

```

## ✨ FeaturesGOOGLE_API_KEY=your-google-api-key

```

### Core Capabilities

3. Run the app:

| Feature | Description |```bash

|---------|-------------|streamlit run app.py

| 📄 **Multimodal Document Processing** | Extract text, images, and tables from PDFs |```

| 🖼️ **True Multimodal Embeddings** | CLIP model embeds both text AND images in unified vector space |

| 🔎 **Hybrid Retrieval** | Combines semantic search (FAISS) with keyword search (BM25) |## Project Structure

| 🎯 **Cross-Encoder Reranking** | Re-scores results for higher precision |

| 💬 **Conversational Memory** | Remembers context across multiple turns |```

| 🔄 **Query Reformulation** | Automatically makes follow-up questions standalone |multimodal_rag_chatbot/

| 📊 **Table Extraction** | Converts PDF tables to searchable markdown |├── app.py                    # Main Streamlit app

| 👁️ **OCR Support** | Extracts text from images using EasyOCR |├── config.py                 # Settings & API keys

├── core/                     # Core processing modules

### User Interface│   ├── document_processor.py # Document loading & chunking

│   ├── embedding_engine.py   # CLIP embeddings + FAISS

| Feature | Description |│   └── retrieval_engine.py   # Hybrid search & reranking

|---------|-------------|├── services/                 # Business logic

| 💻 **Streamlit Chat Interface** | Modern, responsive design |│   ├── llm_service.py        # LLM generation

| 📁 **File Upload** | Drag-and-drop PDF, image, and text files |│   └── rag_pipeline.py       # End-to-end orchestration

| 📚 **Source Citations** | See exactly where answers come from |├── components/               # Streamlit UI components

| ⚡ **Streaming Responses** | Real-time token-by-token output |│   ├── sidebar.py            # File upload sidebar

│   ├── chat_interface.py     # Chat UI

---│   └── document_viewer.py    # Source display

├── data/                     # Data storage

## 🏗️ Architecture│   ├── uploads/              # Uploaded documents

│   ├── faiss_index/          # Vector store

### High-Level System Overview│   └── bm25_index/           # BM25 index

└── utils/                    # Utility functions

```    └── helpers.py

┌─────────────────────────────────────────────────────────────────────┐```

│                         USER INTERFACE                               │
│                      (Streamlit Chat App)                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Query     │  │   Hybrid    │  │  Reranker   │  │    LLM      │ │
│  │Reformulation│→ │  Retrieval  │→ │ (Cross-Enc) │→ │  (Gemini)   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │   FAISS Index     │           │   BM25 Index      │
        │ (Vector Search)   │           │ (Keyword Search)  │
        └───────────────────┘           └───────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DOCUMENT PROCESSING                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │    Text     │  │   Images    │  │   Tables    │  │    OCR      │ │
│  │  Extraction │  │  (+ CLIP)   │  │ (Markdown)  │  │  (EasyOCR)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Multimodal Embedding Architecture

```
                    ┌─────────────────────────────────────┐
                    │         CLIP Model (ViT-B-32)       │
                    │   Shared Text-Image Embedding Space  │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
            ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
            │    Text     │   │   Images    │   │   Tables    │
            │  Encoder    │   │  Encoder    │   │ (as Text)   │
            └─────────────┘   └─────────────┘   └─────────────┘
                    │                 │                 │
                    ▼                 ▼                 ▼
            [512-dim vector]  [512-dim vector]  [512-dim vector]
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                    ┌─────────────────────────────────────┐
                    │        Unified FAISS Index          │
                    │  (Text + Images in same space)      │
                    └─────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Document Processing** | PyMuPDF (fitz) | PDF text, image, table extraction |
| **OCR** | EasyOCR | Extract text from images |
| **Embeddings** | CLIP (ViT-B-32) | Multimodal text & image embeddings |
| **Vector Store** | FAISS | Fast similarity search |
| **Keyword Search** | BM25 (rank-bm25) | Lexical matching |
| **Reranking** | Cross-Encoder | Precision improvement |
| **LLM** | Google Gemini | Response generation |
| **UI Framework** | Streamlit | Web interface |
| **Configuration** | Pydantic Settings | Type-safe config |

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- Google API Key (for Gemini LLM)
- 4GB+ RAM recommended

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd multimodal_rag_chatbot
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional (has defaults)
DEFAULT_MODEL=gemini-2.0-flash
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | - | Google AI API key for Gemini |
| `DEFAULT_MODEL` | ❌ No | `gemini-2.0-flash` | LLM model to use |

### Application Settings

The `config.py` file contains all configurable settings:

```python
class Settings:
    # LLM Settings
    google_api_key: str          # From .env
    default_model: str           # Gemini model name
    
    # Retrieval Settings
    top_k_retrieval: int = 10    # Candidates from hybrid search
    top_k_rerank: int = 3        # Final results after reranking
    
    # Storage Paths
    upload_dir: str = "data/uploads"
    faiss_index_path: str = "data/faiss_index"
    bm25_index_path: str = "data/bm25_index"
```

---

## 🚀 Usage

### Step 1: Upload Documents

1. Click the sidebar file uploader
2. Select PDF, PNG, JPG, or TXT files (multiple files supported)
3. Click **"Process Documents"**
4. Wait for indexing to complete (progress shown)

### Step 2: Ask Questions

Type your question in the chat input:

```
"What are the main points in this document?"
"Summarize the table on page 3"
"What does the diagram show?"
"Explain the methodology section"
```

### Step 3: View Sources

After each response, expand **"📚 Sources"** to see:
- Source file name
- Page number
- Relevant excerpt

### Step 4: Conversation Features

| Action | How To |
|--------|--------|
| **Follow-up Questions** | Just ask "Tell me more" or "Explain that further" |
| **Context Awareness** | System remembers last 5 conversation turns |
| **New Chat** | Click "New Chat" to clear conversation history |
| **Clear All** | Click "Clear All" to reset documents and indices |

---

## 📁 Project Structure

```
multimodal_rag_chatbot/
│
├── app.py                      # 🚀 Streamlit entry point
├── config.py                   # ⚙️ Configuration settings
├── requirements.txt            # 📦 Python dependencies
├── .env                        # 🔐 Environment variables (create this)
├── README.md                   # 📖 This documentation
│
├── core/                       # 🧠 Core processing modules
│   ├── __init__.py
│   ├── document_processor.py   # PDF/image/text extraction, chunking
│   ├── embedding_engine.py     # CLIP embeddings, FAISS index
│   └── retrieval_engine.py     # BM25, hybrid retrieval, reranking
│
├── services/                   # 🔧 Business logic layer
│   ├── __init__.py
│   ├── llm_service.py          # Gemini integration, memory, prompts
│   └── rag_pipeline.py         # Main RAG orchestration
│
├── components/                 # 🎨 UI components
│   ├── __init__.py
│   ├── sidebar.py              # File upload, controls
│   ├── chat_interface.py       # Chat messages display
│   └── document_viewer.py      # Source citations display
│
├── utils/                      # 🛠️ Utility functions
│   ├── __init__.py
│   └── helpers.py              # Helper functions
│
└── data/                       # 💾 Data storage (auto-created)
    ├── uploads/                # Uploaded files
    ├── faiss_index/            # Vector index persistence
    └── bm25_index/             # Keyword index persistence
```

---

## 🔄 How It Works

### 1. Document Processing Pipeline

When you upload a PDF, the system:

```
PDF Upload
    │
    ├──→ TEXT EXTRACTION (PyMuPDF)
    │         │
    │         ▼
    │    Chunking (1000 chars, 200 overlap)
    │    Using RecursiveCharacterTextSplitter
    │         │
    │         ▼
    │    Text Chunks ──→ CLIP Text Encoder ──→ 512-dim Vectors
    │
    ├──→ IMAGE EXTRACTION
    │         │
    │         ├──→ OCR (EasyOCR) ──→ Text content for display
    │         │
    │         └──→ Base64 encoding ──→ Stored in Chunk.image_data
    │                   │
    │                   ▼
    │              CLIP Image Encoder ──→ 512-dim Vectors
    │
    └──→ TABLE EXTRACTION (PyMuPDF)
              │
              ▼
         Convert to Markdown ──→ CLIP Text Encoder ──→ 512-dim Vectors
              
              │
              ▼
    ┌─────────────────────────────────────┐
    │  Unified FAISS Index                │
    │  (All vectors in same semantic      │
    │   space - text queries can find     │
    │   relevant images!)                 │
    │                                     │
    │  + BM25 Keyword Index               │
    │  (For lexical matching)             │
    └─────────────────────────────────────┘
```

### 2. Query Processing Pipeline

When you ask a question:

```
User Question: "What does the chart on page 5 show?"
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  1. QUERY REFORMULATION                                 │
│     ├─ Check conversation history                       │
│     ├─ If follow-up → make standalone using LLM         │
│     └─ "that chart" → "the chart on page 5"             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  2. HYBRID SEARCH                                       │
│     ├─ FAISS Semantic Search (60% weight)               │
│     │   └─ Finds conceptually similar content           │
│     ├─ BM25 Keyword Search (40% weight)                 │
│     │   └─ Finds exact term matches                     │
│     └─ RRF (Reciprocal Rank Fusion) combines results    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  3. CROSS-ENCODER RERANKING                             │
│     ├─ Take top 10 candidates                           │
│     ├─ Score each (query, chunk) pair                   │
│     └─ Return top 3 most relevant                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  4. PROMPT CONSTRUCTION                                 │
│     ├─ System prompt with instructions                  │
│     ├─ Retrieved context with source markers            │
│     ├─ Conversation history (last 2 turns)              │
│     └─ User question                                    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  5. LLM GENERATION (Gemini)                             │
│     ├─ Stream response token by token                   │
│     ├─ Include [Source X] citations                     │
│     └─ Save to conversation memory                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Display in Chat UI with Expandable Sources
```

### 3. Why CLIP for Multimodal Search?

CLIP (Contrastive Language-Image Pre-training) creates a **shared embedding space** for text and images:

```
Text: "solar panel efficiency graph"
                │
                ▼
        CLIP Text Encoder
                │
                ▼
        [0.12, -0.45, 0.89, ..., 0.33]  ← 512-dim vector
                │
                │ CLOSE in vector space (high cosine similarity)
                │
                ▼
        [0.15, -0.42, 0.91, ..., 0.31]  ← 512-dim vector
                │
                ▼
        CLIP Image Encoder
                │
                ▼
Image: 📊 (actual chart of solar panel efficiency)
```

**Benefits:**
- ✅ Text queries can find relevant images
- ✅ Images are searchable by description
- ✅ Unified retrieval across modalities
- ✅ No need for separate image search system

---

## 📚 API Reference

### Core Data Structures

#### `Chunk` (document_processor.py)

The fundamental unit of indexed content:

```python
@dataclass
class Chunk:
    content: str           # Text content or OCR text
    chunk_type: str        # "text" | "table" | "image"
    page_number: int       # Source page number (1-indexed)
    source_file: str       # Source filename
    metadata: dict         # Additional metadata (image dimensions, etc.)
    image_data: str        # Base64-encoded image (for image chunks only)
```

### Core Classes

#### `EmbeddingEngine` (embedding_engine.py)

Handles CLIP embeddings for text and images:

```python
class EmbeddingEngine:
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32")
    
    def embed_text(self, text: str) -> np.ndarray
        """Embed text using CLIP text encoder. Returns 512-dim vector."""
    
    def embed_image(self, image: PIL.Image) -> np.ndarray
        """Embed image using CLIP image encoder. Returns 512-dim vector."""
    
    def embed_batch(self, texts: List[str]) -> np.ndarray
        """Embed multiple texts efficiently."""
    
    def embed_images_batch(self, images: List[PIL.Image]) -> np.ndarray
        """Embed multiple images efficiently."""
```

#### `FAISSIndex` (embedding_engine.py)

Manages the vector store:

```python
class FAISSIndex:
    def __init__(self, dimension: int = 512)
    
    def add_chunks(self, chunks: List[Chunk]) -> None
        """Add chunks to index. Auto-detects text vs image chunks."""
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Chunk, float]]
        """Search for similar chunks. Returns (chunk, score) pairs."""
    
    def save(self, path: str) -> None
        """Persist index to disk."""
    
    def load(self, path: str) -> bool
        """Load index from disk. Returns True if successful."""
    
    def clear(self) -> None
        """Clear all indexed data."""
```

#### `HybridRetriever` (retrieval_engine.py)

Combines FAISS and BM25 search:

```python
class HybridRetriever:
    def __init__(self, faiss_index: FAISSIndex, bm25_index: BM25Index)
    
    def search(self, query: str, k: int = 10, alpha: float = 0.6) -> List[Chunk]
        """
        Hybrid search using RRF fusion.
        alpha: Weight for vector search (1-alpha for BM25)
        """
```

#### `RAGPipeline` (rag_pipeline.py)

Main orchestration class:

```python
class RAGPipeline:
    def __init__(self)
    
    def process_documents(self, file_paths: List[str]) -> int
        """Process and index documents. Returns number of chunks created."""
    
    def query(self, question: str) -> Generator[str, None, None]
        """Run full RAG pipeline. Yields response tokens."""
    
    def get_sources(self) -> List[Chunk]
        """Get source chunks from last query."""
    
    def get_citations_markdown(self) -> str
        """Get formatted citations markdown."""
    
    def clear_memory(self) -> None
        """Clear conversation memory only."""
    
    def clear(self) -> None
        """Clear all indices and reset state."""
```

#### `ConversationMemory` (llm_service.py)

Manages conversation history:

```python
class ConversationMemory:
    def __init__(self, max_turns: int = 5)
    
    def add_user_message(self, content: str) -> None
    def add_assistant_message(self, content: str) -> None
    def get_history_for_reformulation(self) -> str
    def get_history_for_context(self) -> List[Tuple[str, str]]
    def clear(self) -> None
```

### Key Functions

#### `load_document` (document_processor.py)
```python
def load_document(file_path: str) -> List[Chunk]
    """Load and process a document into chunks. Supports PDF, PNG, JPG, TXT."""
```

#### `reformulate_query` (llm_service.py)
```python
def reformulate_query(query: str, memory: ConversationMemory) -> str
    """Make follow-up questions standalone using conversation history."""
```

#### `generate_response` (llm_service.py)
```python
def generate_response(query: str, chunks: List[Chunk], memory: ConversationMemory) -> Generator[str, None, None]
    """Generate streaming LLM response with RAG context."""
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Model Download Timeout

**Error:** `ReadTimeoutError: Read timed out`

**Cause:** Slow network when downloading CLIP or Cross-Encoder models

**Solution:**
```bash
# Pre-download models manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/clip-ViT-B-32')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')"
```

---

#### 2. Corrupted Index Files

**Error:** `EOFError: Ran out of input`

**Cause:** Index files corrupted or from incompatible version

**Solution:**
```powershell
# Delete old indices and reprocess documents
Remove-Item -Recurse -Force data\faiss_index, data\bm25_index
# Then restart app and re-upload documents
```

---

#### 3. Missing API Key

**Error:** `GOOGLE_API_KEY not set` or `Invalid API key`

**Cause:** Missing or incorrect `.env` file

**Solution:**
1. Create `.env` file in project root
2. Add: `GOOGLE_API_KEY=your_actual_key_here`
3. Restart the application

---

#### 4. PNG Color Profile Warning

**Warning:** `libpng warning: iCCP: known incorrect sRGB profile`

**Cause:** PNG images with non-standard color profiles

**Solution:** This is harmless and can be safely ignored. It doesn't affect functionality.

---

#### 5. High Memory Usage

**Cause:** Large documents or many images

**Solutions:**
- Process fewer documents at a time
- Reduce image quality: Change `quality=85` to `quality=60` in `process_pdf()`
- Restart the app to clear memory

---

#### 6. No Results Found

**Cause:** Documents not indexed or query too specific

**Solutions:**
1. Check that "Process Documents" was clicked after upload
2. Try broader search terms
3. Check if documents contain relevant content

---

## 📊 Performance Considerations

| Factor | Recommendation |
|--------|----------------|
| **Document Size** | Best with PDFs under 50 pages |
| **Image Count** | Performance may degrade with 100+ images |
| **Chunk Size** | Default 1000 chars is optimal for most use cases |
| **Memory** | 4GB+ RAM recommended |
| **First Run** | Model download takes 1-5 minutes |

---

## 🔒 Security Notes

1. **API Keys**: Never commit `.env` to version control
2. **Uploaded Files**: Stored locally in `data/uploads/`
3. **Index Data**: Stored locally in `data/` directory
4. **No External Storage**: All data stays on your machine

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - LLM framework
- [Sentence Transformers](https://www.sbert.net/) - CLIP embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search by Meta
- [Streamlit](https://streamlit.io/) - UI framework
- [Google Gemini](https://ai.google.dev/) - LLM
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - OCR engine
