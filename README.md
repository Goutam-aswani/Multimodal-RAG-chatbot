# 🔍 Multimodal RAG Chatbot# 🔍 Multimodal RAG Chatbot# Multimodal RAG Chatbot



A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that processes multimodal documents (PDFs with text, images, and tables) and answers questions using Google's Gemini LLM.



---A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that processes multimodal documents (PDFs with text, images, and tables) and answers questions using Google's Gemini LLM.A Streamlit-based multimodal RAG (Retrieval-Augmented Generation) chatbot that can process PDFs, images, and text files.



## 📋 Table of Contents



- [Features](#-features)---## Features

- [Architecture](#-architecture)

- [Technology Stack](#-technology-stack)

- [Installation](#-installation)

- [Configuration](#-configuration)## 📋 Table of Contents- **Document Processing**: Load PDFs, images, and text files with OCR support

- [Usage](#-usage)

- [Project Structure](#-project-structure)- **Hybrid Search**: Combines FAISS vector search with BM25 keyword search

- [How It Works](#-how-it-works)

- [API Reference](#-api-reference)- [Features](#-features)- **Reranking**: Cross-encoder reranking for improved relevance

- [Troubleshooting](#-troubleshooting)

- [Architecture](#-architecture)- **Streaming Responses**: Real-time LLM response streaming

---

- [Technology Stack](#-technology-stack)- **Citation Tracking**: Source attribution for all responses

## ✨ Features

- [Installation](#-installation)

### Core Capabilities

- [Configuration](#-configuration)## Setup

| Feature | Description |

|---------|-------------|- [Usage](#-usage)

| 📄 **Multimodal Document Processing** | Extract text, images, and tables from PDFs |

| 🖼️ **True Multimodal Embeddings** | CLIP model embeds both text AND images in unified vector space |- [Project Structure](#-project-structure)1. Install dependencies:

| 🔎 **Hybrid Retrieval** | Combines semantic search (FAISS) with keyword search (BM25) |

| 🎯 **Cross-Encoder Reranking** | Re-scores results for higher precision |- [How It Works](#-how-it-works)```bash

| 💬 **Conversational Memory** | Remembers context across multiple turns |

| 🔄 **Query Reformulation** | Automatically makes follow-up questions standalone |- [API Reference](#-api-reference)pip install -r requirements.txt

| 📊 **Table Extraction** | Converts PDF tables to searchable markdown |

| 👁️ **OCR Support** | Extracts text from images using EasyOCR |- [Troubleshooting](#-troubleshooting)```



### User Interface



| Feature | Description |---2. Configure environment variables in `.env`:

|---------|-------------|

| 💻 **Streamlit Chat Interface** | Modern, responsive design |```

| 📁 **File Upload** | Drag-and-drop PDF, image, and text files |

| 📚 **Source Citations** | See exactly where answers come from |## ✨ FeaturesGOOGLE_API_KEY=your-google-api-key

| ⚡ **Streaming Responses** | Real-time token-by-token output |

```

---

### Core Capabilities

## 🏗️ Architecture

3. Run the app:

### High-Level System Overview

| Feature | Description |```bash

```

┌─────────────────────────────────────────────────────────────┐|---------|-------------|streamlit run app.py

│                    Streamlit Web Interface                  │

│  (Chat, File Upload, Source Display, Settings)              │| 📄 **Multimodal Document Processing** | Extract text, images, and tables from PDFs |```

└────────────────────────────┬────────────────────────────────┘

                             │| 🖼️ **True Multimodal Embeddings** | CLIP model embeds both text AND images in unified vector space |

        ┌────────────────────┼────────────────────┐

        │                    │                    │| 🔎 **Hybrid Retrieval** | Combines semantic search (FAISS) with keyword search (BM25) |## Project Structure

        ▼                    ▼                    ▼

┌──────────────┐    ┌────────────────┐   ┌──────────────┐| 🎯 **Cross-Encoder Reranking** | Re-scores results for higher precision |

│   Document   │    │   RAG Pipeline │   │ Conversation │

│  Processor   │    │  & Retrieval   │   │    Memory    │| 💬 **Conversational Memory** | Remembers context across multiple turns |```

│ (PDF, IMG)   │    │   (FAISS+BM25) │   │ (Context)    │

└──────────────┘    └────────────────┘   └──────────────┘| 🔄 **Query Reformulation** | Automatically makes follow-up questions standalone |multimodal_rag_chatbot/

        │                    │                    │

        └────────────────────┼────────────────────┘| 📊 **Table Extraction** | Converts PDF tables to searchable markdown |├── app.py                    # Main Streamlit app

                             │

                    ┌────────▼────────┐| 👁️ **OCR Support** | Extracts text from images using EasyOCR |├── config.py                 # Settings & API keys

                    │  LLM Service    │

                    │  (Gemini API)   │├── core/                     # Core processing modules

                    │  (Streaming)    │

                    └─────────────────┘### User Interface│   ├── document_processor.py # Document loading & chunking

```

│   ├── embedding_engine.py   # CLIP embeddings + FAISS

### Data Flow

| Feature | Description |│   └── retrieval_engine.py   # Hybrid search & reranking

```

1. Document Upload|---------|-------------|├── services/                 # Business logic

   ↓

2. Text Extraction (PyMuPDF) + OCR (EasyOCR) + Table Extraction| 💻 **Streamlit Chat Interface** | Modern, responsive design |│   ├── llm_service.py        # LLM generation

   ↓

3. Chunking (RecursiveCharacterTextSplitter: 1000 chars, 200 overlap)| 📁 **File Upload** | Drag-and-drop PDF, image, and text files |│   └── rag_pipeline.py       # End-to-end orchestration

   ↓

4. Embedding (CLIP: sentence-transformers/clip-ViT-B-32)| 📚 **Source Citations** | See exactly where answers come from |├── components/               # Streamlit UI components

   ├─ Text embeddings (512-dim)

   └─ Image embeddings (512-dim, same space)| ⚡ **Streaming Responses** | Real-time token-by-token output |│   ├── sidebar.py            # File upload sidebar

   ↓

5. Indexing│   ├── chat_interface.py     # Chat UI

   ├─ FAISS (semantic search via cosine similarity)

   └─ BM25 (keyword search)---│   └── document_viewer.py    # Source display

   ↓

6. Query Processing├── data/                     # Data storage

   ├─ Reformulate if follow-up (ConversationMemory)

   ├─ Embed query with CLIP## 🏗️ Architecture│   ├── uploads/              # Uploaded documents

   └─ Hybrid retrieval (RRF fusion)

   ↓│   ├── faiss_index/          # Vector store

7. Reranking (Cross-encoder: ms-marco-MiniLM-L-12-v2)

   ↓### High-Level System Overview│   └── bm25_index/           # BM25 index

8. Generation (Gemini with streaming)

   ↓└── utils/                    # Utility functions

9. Display with source citations

``````    └── helpers.py



---┌─────────────────────────────────────────────────────────────────────┐```



## 💻 Technology Stack│                         USER INTERFACE                               │

│                      (Streamlit Chat App)                            │

| Component | Technology |└─────────────────────────────────────────────────────────────────────┘

|-----------|-----------|                                    │

| **LLM** | Google Gemini API |                                    ▼

| **Embeddings** | CLIP (sentence-transformers) |┌─────────────────────────────────────────────────────────────────────┐

| **Vector DB** | FAISS (IndexFlatIP) |│                         RAG PIPELINE                                 │

| **Keyword Search** | BM25 (rank-bm25) |│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │

| **Reranking** | Cross-Encoder (sentence-transformers) |│  │   Query     │  │   Hybrid    │  │  Reranker   │  │    LLM      │ │

| **PDF Processing** | PyMuPDF (fitz) |│  │Reformulation│→ │  Retrieval  │→ │ (Cross-Enc) │→ │  (Gemini)   │ │

| **OCR** | EasyOCR |│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │

| **UI Framework** | Streamlit |└─────────────────────────────────────────────────────────────────────┘

| **LLM Integration** | LangChain (langchain-google-genai) |                                    │

| **Text Splitting** | LangChain TextSplitters |                    ┌───────────────┴───────────────┐

| **Python Version** | 3.8+ |                    ▼                               ▼

        ┌───────────────────┐           ┌───────────────────┐

---        │   FAISS Index     │           │   BM25 Index      │

        │ (Vector Search)   │           │ (Keyword Search)  │

## 📦 Installation        └───────────────────┘           └───────────────────┘

                    │                               │

### Prerequisites                    └───────────────┬───────────────┘

- Python 3.8 or higher                                    ▼

- Google API key (free at [ai.google.dev](https://ai.google.dev))┌─────────────────────────────────────────────────────────────────────┐

│                    DOCUMENT PROCESSING                               │

### Steps│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │

│  │    Text     │  │   Images    │  │   Tables    │  │    OCR      │ │

1. **Clone the repository**│  │  Extraction │  │  (+ CLIP)   │  │ (Markdown)  │  │  (EasyOCR)  │ │

```bash│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │

git clone https://github.com/Goutam-aswani/Multimodal-RAG-chatbot.git└─────────────────────────────────────────────────────────────────────┘

cd Multimodal-RAG-chatbot```

```

### Multimodal Embedding Architecture

2. **Create virtual environment**

```bash```

python -m venv venv                    ┌─────────────────────────────────────┐

source venv/bin/activate  # On Windows: venv\Scripts\activate                    │         CLIP Model (ViT-B-32)       │

```                    │   Shared Text-Image Embedding Space  │

                    └─────────────────────────────────────┘

3. **Install dependencies**                                      │

```bash                    ┌─────────────────┼─────────────────┐

pip install -r requirements.txt                    │                 │                 │

```                    ▼                 ▼                 ▼

            ┌─────────────┐   ┌─────────────┐   ┌─────────────┐

4. **Create `.env` file**            │    Text     │   │   Images    │   │   Tables    │

```bash            │  Encoder    │   │  Encoder    │   │ (as Text)   │

echo "GOOGLE_API_KEY=your_api_key_here" > .env            └─────────────┘   └─────────────┘   └─────────────┘

```                    │                 │                 │

                    ▼                 ▼                 ▼

5. **Run the app**            [512-dim vector]  [512-dim vector]  [512-dim vector]

```bash                    │                 │                 │

streamlit run app.py                    └─────────────────┼─────────────────┘

```                                      ▼

                    ┌─────────────────────────────────────┐

The app will open at `http://localhost:8501`                    │        Unified FAISS Index          │

                    │  (Text + Images in same space)      │

---                    └─────────────────────────────────────┘

```

## ⚙️ Configuration

---

### Environment Variables (`.env`)

## 🛠️ Technology Stack

```env

# Required| Component | Technology | Purpose |

GOOGLE_API_KEY=your-gemini-api-key|-----------|------------|---------|

| **Document Processing** | PyMuPDF (fitz) | PDF text, image, table extraction |

# Optional| **OCR** | EasyOCR | Extract text from images |

CHUNK_SIZE=1000              # Text chunk size| **Embeddings** | CLIP (ViT-B-32) | Multimodal text & image embeddings |

CHUNK_OVERLAP=200            # Chunk overlap| **Vector Store** | FAISS | Fast similarity search |

MAX_MEMORY_TURNS=5           # Conversation history turns| **Keyword Search** | BM25 (rank-bm25) | Lexical matching |

TOP_K_RETRIEVAL=10           # Top K results to retrieve| **Reranking** | Cross-Encoder | Precision improvement |

RERANK_TOP_K=5               # Top K results after reranking| **LLM** | Google Gemini | Response generation |

```| **UI Framework** | Streamlit | Web interface |

| **Configuration** | Pydantic Settings | Type-safe config |

### Supported File Types

---

| Type | Extensions | Processing |

|------|-----------|-----------|## 📦 Installation

| **PDF** | `.pdf` | Text, images, tables |

| **Images** | `.jpg`, `.png`, `.jpeg` | OCR + CLIP embedding |### Prerequisites

| **Text** | `.txt`, `.md` | Direct text processing |

- Python 3.9 or higher

---- Google API Key (for Gemini LLM)

- 4GB+ RAM recommended

## 🚀 Usage

### Step 1: Clone the Repository

### Quick Start

```bash

1. **Upload documents** - Click "Upload Documents" in the sidebargit clone <repository-url>

2. **Process** - Click "Process Documents" buttoncd multimodal_rag_chatbot

3. **Ask questions** - Type in the chat input```

4. **View sources** - Expand "View Sources" for context

### Step 2: Create Virtual Environment

### Example Queries

```bash

- "What is the main topic of this document?"# Create virtual environment

- "Summarize the key points"python -m venv venv

- "What does the image show?"

- "Can you explain the table?"# Activate (Windows)

- "Follow-up: Tell me more about..." (uses conversation memory)venv\Scripts\activate



---# Activate (Linux/Mac)

source venv/bin/activate

## 📁 Project Structure```



```### Step 3: Install Dependencies

multimodal_rag_chatbot/

├── app.py                      # Main Streamlit app```bash

├── config.py                   # Settings & configurationpip install -r requirements.txt

├── requirements.txt            # Python dependencies```

├── .env                        # Environment variables

├── .gitignore                  # Git ignore file### Step 4: Configure Environment

├── README.md                   # This file

├── QUICKSTART.md               # Quick start guideCreate a `.env` file in the project root:

│

├── core/                       # Core processing modules```env

│   ├── document_processor.py   # Document loading & chunking# Required

│   ├── embedding_engine.py     # CLIP embeddings + FAISSGOOGLE_API_KEY=your_google_api_key_here

│   └── retrieval_engine.py     # Hybrid search & reranking

│# Optional (has defaults)

├── services/                   # Business logicDEFAULT_MODEL=gemini-2.0-flash

│   ├── llm_service.py         # LLM generation & memory```

│   └── rag_pipeline.py        # End-to-end orchestration

│### Step 5: Run the Application

├── components/                 # Streamlit UI components

│   ├── sidebar.py             # File upload sidebar```bash

│   ├── chat_interface.py       # Chat UIstreamlit run app.py

│   └── document_viewer.py      # Source display```

│

└── data/                       # Data storageThe app will open at `http://localhost:8501`

    ├── uploads/               # Uploaded documents

    ├── faiss_index/           # Vector store---

    └── bm25_index/            # BM25 index

```## ⚙️ Configuration



---### Environment Variables



## 🔄 How It Works| Variable | Required | Default | Description |

|----------|----------|---------|-------------|

### 1. Document Processing Pipeline| `GOOGLE_API_KEY` | ✅ Yes | - | Google AI API key for Gemini |

| `DEFAULT_MODEL` | ❌ No | `gemini-2.0-flash` | LLM model to use |

```python

# Load document### Application Settings

doc = process_pdf("document.pdf")

The `config.py` file contains all configurable settings:

# Extract: text, images (as base64), tables (as markdown)

# Chunk: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)```python

# Store: Chunk dataclass with metadataclass Settings:

```    # LLM Settings

    google_api_key: str          # From .env

### 2. Embedding & Indexing    default_model: str           # Gemini model name

    

```python    # Retrieval Settings

# CLIP embeddings (unified space)    top_k_retrieval: int = 10    # Candidates from hybrid search

text_embedding = clip.embed_text(text)      # 512-dim vector    top_k_rerank: int = 3        # Final results after reranking

image_embedding = clip.embed_image(image)   # 512-dim vector (same space!)    

    # Storage Paths

# Store in FAISS    upload_dir: str = "data/uploads"

faiss_index.add(embeddings)  # Cosine similarity via L2 normalization    faiss_index_path: str = "data/faiss_index"

    bm25_index_path: str = "data/bm25_index"

# Store in BM25```

bm25_index.add_document(text)  # Keyword-based ranking

```---



### 3. Retrieval## 🚀 Usage



```python### Step 1: Upload Documents

# Reformulate query if follow-up

query = reformulate_query_if_needed(user_input, memory)1. Click the sidebar file uploader

2. Select PDF, PNG, JPG, or TXT files (multiple files supported)

# Hybrid retrieval3. Click **"Process Documents"**

faiss_results = faiss_index.search(query_embedding, top_k=10)4. Wait for indexing to complete (progress shown)

bm25_results = bm25_index.search(query, top_k=10)

### Step 2: Ask Questions

# Fuse with RRF (Reciprocal Rank Fusion)

final_results = rrf_fusion(faiss_results, bm25_results)Type your question in the chat input:



# Rerank with cross-encoder```

reranked = reranker.rank(final_results, top_k=5)"What are the main points in this document?"

```"Summarize the table on page 3"

"What does the diagram show?"

### 4. Generation"Explain the methodology section"

```

```python

# Build context from retrieved chunks### Step 3: View Sources

context = "\n".join([chunk.text for chunk in reranked])

After each response, expand **"📚 Sources"** to see:

# Generate with Gemini (streaming)- Source file name

response = gemini.generate(context, user_query, memory)- Page number

- Relevant excerpt

# Update memory

memory.add_turn(user_query, response)### Step 4: Conversation Features

```

| Action | How To |

---|--------|--------|

| **Follow-up Questions** | Just ask "Tell me more" or "Explain that further" |

## 📚 API Reference| **Context Awareness** | System remembers last 5 conversation turns |

| **New Chat** | Click "New Chat" to clear conversation history |

### Core Modules| **Clear All** | Click "Clear All" to reset documents and indices |



#### `document_processor.py`---



```python## 📁 Project Structure

from core.document_processor import process_pdf, process_image

```

# Process PDFmultimodal_rag_chatbot/

chunks = process_pdf("document.pdf")│

# Returns: List[Chunk] with text, images (base64), metadata├── app.py                      # 🚀 Streamlit entry point

├── config.py                   # ⚙️ Configuration settings

# Process Image├── requirements.txt            # 📦 Python dependencies

chunks = process_image("image.png")├── .env                        # 🔐 Environment variables (create this)

# Returns: List[Chunk] with image data and metadata├── README.md                   # 📖 This documentation

```│

├── core/                       # 🧠 Core processing modules

#### `embedding_engine.py`│   ├── __init__.py

│   ├── document_processor.py   # PDF/image/text extraction, chunking

```python│   ├── embedding_engine.py     # CLIP embeddings, FAISS index

from core.embedding_engine import EmbeddingEngine│   └── retrieval_engine.py     # BM25, hybrid retrieval, reranking

│

engine = EmbeddingEngine(model_name="sentence-transformers/clip-ViT-B-32")├── services/                   # 🔧 Business logic layer

│   ├── __init__.py

# Embed text│   ├── llm_service.py          # Gemini integration, memory, prompts

embedding = engine.embed_text("Hello world")  # 512-dim vector│   └── rag_pipeline.py         # Main RAG orchestration

│

# Embed image├── components/                 # 🎨 UI components

embedding = engine.embed_image(image_bytes)   # 512-dim vector (same space)│   ├── __init__.py

│   ├── sidebar.py              # File upload, controls

# Add chunks to FAISS│   ├── chat_interface.py       # Chat messages display

engine.faiss_index.add_chunks(chunks)│   └── document_viewer.py      # Source citations display

```│

├── utils/                      # 🛠️ Utility functions

#### `retrieval_engine.py`│   ├── __init__.py

│   └── helpers.py              # Helper functions

```python│

from core.retrieval_engine import HybridRetriever└── data/                       # 💾 Data storage (auto-created)

    ├── uploads/                # Uploaded files

retriever = HybridRetriever(faiss_index, bm25_index, reranker)    ├── faiss_index/            # Vector index persistence

    └── bm25_index/             # Keyword index persistence

# Retrieve relevant chunks```

results = retriever.retrieve(query_embedding, query_text, top_k=5)

# Returns: List[Chunk] ranked by relevance---

```

## 🔄 How It Works

#### `rag_pipeline.py`

### 1. Document Processing Pipeline

```python

from services.rag_pipeline import RAGPipelineWhen you upload a PDF, the system:



pipeline = RAGPipeline()```

PDF Upload

# Process documents    │

pipeline.process_documents(file_paths)    ├──→ TEXT EXTRACTION (PyMuPDF)

    │         │

# Query with streaming    │         ▼

for chunk in pipeline.query("What is this about?"):    │    Chunking (1000 chars, 200 overlap)

    print(chunk, end="", flush=True)    │    Using RecursiveCharacterTextSplitter

```    │         │

    │         ▼

#### `llm_service.py`    │    Text Chunks ──→ CLIP Text Encoder ──→ 512-dim Vectors

    │

```python    ├──→ IMAGE EXTRACTION

from services.llm_service import LLMService, ConversationMemory    │         │

    │         ├──→ OCR (EasyOCR) ──→ Text content for display

memory = ConversationMemory(max_turns=5)    │         │

llm = LLMService()    │         └──→ Base64 encoding ──→ Stored in Chunk.image_data

    │                   │

# Generate response    │                   ▼

response = llm.generate_response(    │              CLIP Image Encoder ──→ 512-dim Vectors

    context="...",    │

    query="Question?",    └──→ TABLE EXTRACTION (PyMuPDF)

    memory=memory,              │

    stream=True              ▼

)         Convert to Markdown ──→ CLIP Text Encoder ──→ 512-dim Vectors

```              

              │

---              ▼

    ┌─────────────────────────────────────┐

## 🐛 Troubleshooting    │  Unified FAISS Index                │

    │  (All vectors in same semantic      │

### Common Issues    │   space - text queries can find     │

    │   relevant images!)                 │

| Problem | Solution |    │                                     │

|---------|----------|    │  + BM25 Keyword Index               │

| **"API key not found"** | Check `.env` file exists and has correct `GOOGLE_API_KEY` |    │  (For lexical matching)             │

| **"Module not found"** | Run `pip install -r requirements.txt` |    └─────────────────────────────────────┘

| **Slow first run** | Models downloading (CLIP, cross-encoder) — one-time only |```

| **Out of memory** | Reduce `CHUNK_SIZE` or `MAX_RETRIEVAL_RESULTS` in `config.py` |

| **"FAISS index not found"** | Process documents first using the sidebar button |### 2. Query Processing Pipeline

| **PDF not extracting text** | Check PDF is not scanned image; use OCR (auto-enabled for images) |

| **Embedding dimension mismatch** | Ensure same CLIP model used for all embeddings |When you ask a question:



### Debug Mode```

User Question: "What does the chart on page 5 show?"

Enable debug logging in `config.py`:    │

```python    ▼

DEBUG = True┌─────────────────────────────────────────────────────────┐

LOGGING_LEVEL = "DEBUG"│  1. QUERY REFORMULATION                                 │

```│     ├─ Check conversation history                       │

│     ├─ If follow-up → make standalone using LLM         │

---│     └─ "that chart" → "the chart on page 5"             │

└─────────────────────────────────────────────────────────┘

## 📝 License    │

    ▼

MIT License - see LICENSE file for details┌─────────────────────────────────────────────────────────┐

│  2. HYBRID SEARCH                                       │

---│     ├─ FAISS Semantic Search (60% weight)               │

│     │   └─ Finds conceptually similar content           │

## 🤝 Contributing│     ├─ BM25 Keyword Search (40% weight)                 │

│     │   └─ Finds exact term matches                     │

Contributions welcome! Please:│     └─ RRF (Reciprocal Rank Fusion) combines results    │

└─────────────────────────────────────────────────────────┘

1. Fork the repository    │

2. Create a feature branch (`git checkout -b feature/amazing-feature`)    ▼

3. Commit changes (`git commit -m 'Add amazing feature'`)┌─────────────────────────────────────────────────────────┐

4. Push to branch (`git push origin feature/amazing-feature`)│  3. CROSS-ENCODER RERANKING                             │

5. Open a Pull Request│     ├─ Take top 10 candidates                           │

│     ├─ Score each (query, chunk) pair                   │

---│     └─ Return top 3 most relevant                       │

└─────────────────────────────────────────────────────────┘

## 👨‍💻 Author    │

    ▼

**Goutam Aswani**┌─────────────────────────────────────────────────────────┐

│  4. PROMPT CONSTRUCTION                                 │

---│     ├─ System prompt with instructions                  │

│     ├─ Retrieved context with source markers            │

## 🙏 Acknowledgments│     ├─ Conversation history (last 2 turns)              │

│     └─ User question                                    │

- [Google Gemini API](https://ai.google.dev)└─────────────────────────────────────────────────────────┘

- [CLIP (OpenAI)](https://github.com/openai/CLIP)    │

- [FAISS (Meta)](https://github.com/facebookresearch/faiss)    ▼

- [LangChain](https://langchain.com)┌─────────────────────────────────────────────────────────┐

- [Streamlit](https://streamlit.io)│  5. LLM GENERATION (Gemini)                             │

│     ├─ Stream response token by token                   │

---│     ├─ Include [Source X] citations                     │

│     └─ Save to conversation memory                      │

**Made with ❤️ for multimodal AI**└─────────────────────────────────────────────────────────┘

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
