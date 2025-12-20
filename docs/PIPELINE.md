# 🔄 Complete Pipeline Workflow

This document explains the entire workflow of the Multimodal RAG Chatbot, from document processing through response display.

---

## 📋 Table of Contents

1. [Document Processing Pipeline](#document-processing-pipeline)
2. [Query Processing Pipeline](#query-processing-pipeline)
3. [Final Query Construction](#final-query-construction)
4. [Response Generation](#response-generation)
5. [Response Display](#response-display)
6. [Complete End-to-End Flow](#complete-end-to-end-flow)

---

## Document Processing Pipeline

### Overview

When a user uploads documents (PDFs, images, or text files), the system processes them:

```
User Uploads Files
    │
    ├─ PDF File → Extract text, images, tables
    ├─ Image File → Load and encode to base64 + OCR
    └─ Text File → Read and chunk text
    │
    ▼
All chunks with metadata
    │
    ▼
CLIP Embedding (Unified vector space)
    │
    ├─ Text chunks → CLIP Text Encoder → 512-dim
    ├─ Image chunks → CLIP Image Encoder → 512-dim
    └─ Table chunks → CLIP Text Encoder → 512-dim
    │
    ▼
Store in FAISS Index + BM25 Index
```

### PDF Processing Steps

1. **Text Extraction** (PyMuPDF)
   - Extract text from each page
   - Split into chunks (1000 chars, 200 overlap)
   - Create Chunk objects with metadata

2. **Image Extraction** (PyMuPDF)
   - Extract image bytes
   - Convert to PIL Image
   - Encode to Base64 (for storage)
   - Run OCR (EasyOCR) for description
   - Create image Chunk with OCR text + base64 data

3. **Table Extraction** (PyMuPDF)
   - Find tables in page
   - Convert to Markdown format
   - Create table Chunk with Markdown content

---

## Query Processing Pipeline

### Overview

```
User Query
    │
    ▼
1. Reformulation (Check conversation history)
    │
    ▼
2. Embedding (CLIP Text Encoder)
    │
    ▼
3. Hybrid Retrieval (FAISS + BM25 + RRF)
    │
    ▼
4. Reranking (Cross-Encoder)
    │
    ▼
5. Build Context (Top results)
    │
    ▼
6. Construct Final Prompt
    │
    ▼
Send to Gemini API
```

### Step-by-Step Details

#### Step 1: Query Reformulation
```
Check conversation memory:
- If follow-up question → Reformulate using LLM to make standalone
- If new question → Use as-is

Example:
User: "Tell me more about the chart"
Memory: "Previous: What are the main findings?"
Result: "Tell me more about the chart from the main findings mentioned in the report"
```

#### Step 2: Query Embedding
```
Reformulated query → CLIP Text Encoder → 512-dim vector
Example: [0.12, -0.45, 0.89, ..., 0.33]
```

#### Step 3: Hybrid Retrieval
```
A) FAISS Semantic Search (60% weight)
   Query vector → Search FAISS → Top 10 semantic matches
   
B) BM25 Keyword Search (40% weight)
   Query text → Search BM25 → Top 10 keyword matches
   
C) Reciprocal Rank Fusion (RRF)
   Combine scores → Final ranking → Top 10 candidates
```

#### Step 4: Reranking
```
For each of 10 candidates:
   Cross-Encoder(query, chunk) → Relevance score
   
Sort by score → Keep top 3 results
```

---

## Final Query Construction

### What Gets Sent to Gemini API

```
FINAL PROMPT = System Prompt + Context + History + Query

System Prompt:
"You are a helpful AI assistant that answers questions based on provided context.
Always cite your sources using [Source X] format. Be accurate and concise."

+

Context (from top 3 reranked results):
"[Source 1: report.pdf, Page 1]
Main findings include a 25% increase in sales compared to last quarter...

[Source 2: report.pdf, Page 3]
The bar chart shows market trends with strong growth in Q4...

[Source 3: report.pdf, Page 5]
Key recommendations: Expand team, increase marketing budget..."

+

Conversation History (last 2 turns from memory):
"User: What are the main findings?
Assistant: The report shows three main findings: 1) 25% sales increase, 2) Market growth, 3) Key recommendations.

User: Tell me more about the chart."

+

User Query (reformulated):
"Tell me more about the chart showing sales data from the report"
```

This complete prompt is sent to Gemini API with streaming enabled.

---

## Response Generation

### How Gemini Responds

```
Gemini API receives complete prompt
    │
    ▼
Process with LLM (gemini-2.0-flash)
    │
    ├─ Read context and understand query
    ├─ Generate response
    ├─ Include source citations
    └─ Stream tokens back
    │
    ▼
Response Stream (token by token)
"The" → "chart" → "shows" → "market" → "trends"...
```

### Response Structure

```
Response from Gemini:
"The chart shows market trends with strong growth in Q4. Based on the data 
visualization, we can see:

1. **Q4 Growth**: Sales increased significantly
2. **Trend Analysis**: Upward trajectory suggests sustainable growth
3. **Market Position**: Position strengthened compared to competitors

[Source 1: report.pdf, Page 3]
[Source 2: report.pdf, Page 1]
[Source 3: report.pdf, Page 5]"
```

---

## Response Display

### Streamlit UI Display

```
┌────────────────────────────────────────┐
│      STREAMLIT CHAT INTERFACE          │
│                                        │
│ User:                                  │
│ "Tell me more about the chart..."     │
│                                        │
│ Assistant (Streaming):                 │
│ "The chart shows market trends with   │
│  strong growth in Q4..."               │
│                                        │
│ 📚 View Sources (Expandable)          │
│   Source 1: report.pdf, Page 3        │
│   "The bar chart shows..."             │
│                                        │
│   Source 2: report.pdf, Page 1        │
│   "Main findings include..."           │
│                                        │
└────────────────────────────────────────┘
```

### Code Flow

```python
# Display user message in chat
with st.chat_message("user"):
    st.write(user_query)

# Display assistant response (streaming)
with st.chat_message("assistant"):
    response_placeholder = st.empty()
    full_response = ""
    
    for chunk in response:  # Stream from Gemini
        if chunk.text:
            full_response += chunk.text
            response_placeholder.markdown(full_response)  # Real-time update
    
# Display source citations
with st.expander("📚 View Sources"):
    for i, source in enumerate(retrieved_sources, 1):
        st.write(f"**Source {i}: {source.source_file}, Page {source.page_number}**")
        st.write(source.content[:300] + "...")

# Save to conversation memory for next turn
conversation_memory.add_user_message(user_query)
conversation_memory.add_assistant_message(full_response)
```

---

## Complete End-to-End Flow

### Full Journey

```
┌────────────────────────────────────────────────────────────┐
│  1. DOCUMENT UPLOAD & PROCESSING                           │
├────────────────────────────────────────────────────────────┤
│ Input: "report.pdf"                                        │
│ Processing:                                                │
│ ├─ Extract 200 text chunks                                │
│ ├─ Extract 5 images with base64 + OCR                     │
│ ├─ Extract 3 tables as markdown                           │
│ └─ Total: 208 Chunk objects                               │
│ Embedding:                                                 │
│ ├─ Text → CLIP Text → 512-dim vectors                     │
│ ├─ Images → CLIP Image → 512-dim vectors                  │
│ └─ Tables → CLIP Text → 512-dim vectors                   │
│ Storage:                                                   │
│ ├─ FAISS Index: 208 vectors                               │
│ └─ BM25 Index: 208 documents                              │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│  2. USER QUERIES                                           │
├────────────────────────────────────────────────────────────┤
│ Query: "What are the main findings?"                       │
│                                                            │
│ Processing:                                               │
│ ├─ No previous context → Use query as-is                  │
│ ├─ Embed query → 512-dim vector                           │
│ ├─ FAISS search (k=10) + BM25 search (k=10)              │
│ ├─ RRF fusion → Top 10 candidates                         │
│ ├─ Cross-Encoder reranking → Top 3 results               │
│ └─ Selected chunks ready for LLM                          │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│  3. PROMPT CONSTRUCTION                                    │
├────────────────────────────────────────────────────────────┤
│ Final Prompt:                                              │
│ ├─ System: "You are a helpful assistant..."              │
│ ├─ Context: [Source 1], [Source 2], [Source 3]           │
│ ├─ History: (empty for first query)                       │
│ └─ Query: "What are the main findings?"                   │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│  4. GEMINI API RESPONSE (STREAMING)                        │
├────────────────────────────────────────────────────────────┤
│ Token stream:                                              │
│ "The" "report" "shows" "three" "main" "findings"...       │
│                                                            │
│ Full Response:                                             │
│ "The report shows three main findings:                    │
│  1. Sales increased 25%                                   │
│  2. Market growth evident                                 │
│  3. Key recommendations                                   │
│  [Source 1], [Source 2], [Source 3]"                      │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│  5. DISPLAY IN STREAMLIT                                   │
├────────────────────────────────────────────────────────────┤
│ ├─ User message displayed                                 │
│ ├─ Assistant response displayed (real-time streaming)     │
│ ├─ Source citations visible                               │
│ ├─ "View Sources" expandable section                      │
│ └─ Response saved to memory                               │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│  6. MEMORY UPDATE FOR NEXT TURN                            │
├────────────────────────────────────────────────────────────┤
│ ConversationMemory:                                        │
│ ├─ Turn 1:                                                │
│ │  User: "What are the main findings?"                    │
│ │  Assistant: "The report shows three main findings..."   │
│ │                                                         │
│ └─ Ready for follow-up questions                          │
│    (Next query will use this history for reformulation)   │
└────────────────────────────────────────────────────────────┘
```

---

## Key Components Map

| Component | File | Function | Input → Output |
|-----------|------|----------|-----------------|
| Document Loading | `document_processor.py` | `load_document()` | PDF/Image/Text → List[Chunk] |
| Text Extraction | `document_processor.py` | `process_pdf()` | PDF → Text chunks |
| Image Processing | `document_processor.py` | `process_image()` | Image → Chunk with base64 |
| OCR | `document_processor.py` | `run_ocr()` | PIL Image → Text string |
| Embedding | `embedding_engine.py` | `embed_text()`, `embed_image()` | Text/Image → 512-dim vector |
| FAISS Indexing | `embedding_engine.py` | `FAISSIndex` | Vectors → Index |
| BM25 Indexing | `retrieval_engine.py` | `BM25Index` | Text → Index |
| Query Reformulation | `llm_service.py` | `reformulate_query()` | Query + Memory → Standalone query |
| Hybrid Search | `retrieval_engine.py` | `HybridRetriever` | Query → Top 10 candidates |
| Reranking | `retrieval_engine.py` | `Reranker` | Query + Candidates → Top 3 |
| Prompt Building | `rag_pipeline.py` | `query()` | Context + Query → Final prompt |
| LLM Response | Google Gemini API | N/A | Prompt → Streaming response |
| Display | `components/chat_interface.py` | Streamlit widgets | Response → Chat UI |
| Memory | `llm_service.py` | `ConversationMemory` | Query + Response → Updated memory |

---

## Data Flow Diagram

```
User Input
    │
    ├─→ Document Upload?
    │   └─→ document_processor.py
    │       └─→ Extract + Chunk → embedding_engine.py
    │           └─→ FAISS + BM25 Indices
    │
    └─→ Query Input
        │
        ├─→ llm_service.py (Reformulation)
        │   └─→ ConversationMemory
        │
        ├─→ embedding_engine.py (Embed)
        │   └─→ 512-dim query vector
        │
        ├─→ retrieval_engine.py (Search)
        │   ├─→ FAISS (semantic)
        │   ├─→ BM25 (keyword)
        │   └─→ RRF Fusion → Top 10
        │
        ├─→ retrieval_engine.py (Rerank)
        │   └─→ Cross-Encoder → Top 3
        │
        ├─→ rag_pipeline.py (Prompt)
        │   └─→ Final prompt construction
        │
        ├─→ Google Gemini API
        │   └─→ Streaming response
        │
        ├─→ llm_service.py (Memory)
        │   └─→ Save to ConversationMemory
        │
        └─→ Streamlit (Display)
            ├─→ User message
            ├─→ Assistant response (streaming)
            ├─→ Source citations
            └─→ Expandable sources
```

---

## Performance Characteristics

- **Document Processing**: ~100 pages/minute
- **Query to Response**: 2-5 seconds (including API latency)
- **Memory Usage**: 2GB baseline + 1GB per 1000 chunks
- **Vector Similarity**: Cosine (L2 normalized in FAISS)
- **Embedding Dimension**: 512-dim (CLIP ViT-B-32)
- **Max Conversation Turns**: 5 (in memory)
- **Retrieved Context Size**: Top 3 chunks
- **Reranking Speed**: ~50ms per query

---

**Document Version**: 1.0  
**Created**: December 20, 2025
