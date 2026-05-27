# Medical-Chatbot - Architecture & Rebuild Guide

## Current Architecture Overview

```
Medical-Chatbot (3-Tier Architecture)
├── Presentation Layer
│   ├── Flask Web Server (8080)
│   └── Chat HTML Interface (Bootstrap + jQuery)
│
├── Business Logic Layer
│   ├── RAG Pipeline (LangChain)
│   ├── Document Processing (src/)
│   └── Prompt Management
│
└── Data & Infrastructure Layer
    ├── Vector Database (Pinecone)
    ├── LLM API (Groq)
    └── Embeddings Model (HuggingFace)
```

---

## Component Breakdown

### 1. **Frontend (Presentation Layer)**
```
chat.html
├── Structure (HTML)
│   └── Chat container, message area, input form
├── Styling (CSS in style.css)
│   └── Modern gradient, responsive, animations
└── Logic (Inline JavaScript)
    ├── Form submission handling
    ├── Message rendering
    ├── AJAX communication
    └── XSS prevention
```

**Characteristics**:
- Simple, lightweight (no framework)
- Single-page application (no routing)
- Real-time messaging simulation
- Works offline (no dependency on backend for UI)

---

### 2. **Backend (Business Logic Layer)**

#### `app.py` - Web Server
```python
╔═══════════════════════════════════════════╗
║      Flask Application (app.py)           ║
╠═══════════════════════════════════════════╣
║  Routes:                                  ║
║  ├── GET  / → Renders chat.html           ║
║  └── POST /get → Processes user query     ║
║                                           ║
║  Services:                                ║
║  ├── Initialize embeddings                ║
║  ├── Connect to Pinecone                  ║
║  ├── Create RAG pipeline                  ║
║  └── Handle HTTP requests                 ║
╚═══════════════════════════════════════════╝
```

#### `src/helper.py` - Document Processing
```python
Functions:
├── load_pdf_documents()      → Read PDFs from directory
├── clean_document_metadata() → Remove unnecessary fields
├── split_documents_into_chunks() → Text chunking with overlap
└── initialize_embeddings()   → Load HuggingFace model
```

**Flow**:
```
PDF Files → Loaded → Cleaned → Split into Chunks → Convert to Embeddings
```

#### `src/prompt.py` - Prompt Engineering
```python
get_system_prompt_template()
    ↓
Returns: System instruction for LLM context
- Medical assistant role definition
- Context instruction
- Output format specification (3 sentences max)
```

#### `store_index.py` - Data Pipeline
```python
Main function orchestrates:
1. PDF loading
2. Metadata cleaning
3. Text chunking
4. Embedding generation
5. Pinecone index creation
6. Batch upload to vector store
```

---

### 3. **Data & Infrastructure Layer**

#### Vector Database (Pinecone)
```
Organization:
├── Index: "medical-chatbot"
│   └── Vectors (384-dimensional)
│       ├── Document chunk 1 + embedding
│       ├── Document chunk 2 + embedding
│       └── Document chunk N + embedding
│
Query Operation:
├── User query → Convert to 384-d vector
├── Search in Pinecone (k=3 similarity search)
└── Return top 3 matching document chunks
```

#### LLM Service (Groq API)
```
Endpoint: Groq API
├── Model: llama-3.1-8b-instant
├── Temperature: 0 (deterministic)
├── Max tokens: Default (varies by model)
└── Input: System prompt + Retrieved documents + User query
```

#### Embeddings (HuggingFace)
```
Model: sentence-transformers/all-MiniLM-L6-v2
├── Output dimension: 384
├── Speed: Fast inference
├── Quality: Good semantic understanding
└── Type: Sentence embeddings (ideal for RAG)
```

---

## RAG Pipeline Deep Dive

### Request Flow
```
User Input (Chat Interface)
    ↓
HTTP POST /get
    ↓ ┌─────────────────────────────────────────────┐
    └→│ app.py: get_response()                      │
      │ 1. Get user message from form               │
      │ 2. Input validation                         │
      │ 3. Check cache (optional)                   │
      └─────────────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────────────┐
    │ RAG Pipeline: rag_pipeline.invoke()         │
    │ (LangChain's create_retrieval_chain)        │
    │                                             │
    │ Step 1: Vector Search in Pinecone          │
    │    ├─ Convert user query to 384-d vector   │
    │    └─ Find top 3 similar documents         │
    │                                             │
    │ Step 2: Prepare Context                    │
    │    ├─ Combine system prompt                │
    │    ├─ Add retrieved documents              │
    │    └─ Add user query                       │
    │                                             │
    │ Step 3: LLM Generation                     │
    │    ├─ Send to Groq API                     │
    │    ├─ Llama 3.1 8B processes               │
    │    └─ Returns response (3 sentences)       │
    │                                             │
    └─────────────────────────────────────────────┘
        ↓
    Response String
        ↓
    Return to Frontend (AJAX)
        ↓
    Display in Chat Interface
```

### Key Parameters (Tunable)
```
RETRIEVAL_K = 3              # Number of documents to retrieve
                             # (Higher = more context, slower)

CHUNK_SIZE = 500            # Characters per chunk
                             # (Larger = less chunking overhead, 
                             #  less granular search)

TEMPERATURE = 0             # Model randomness (0 = deterministic)
                             # (For medical, determinism is critical)

EMBEDDING_DIM = 384         # Vector dimension (fixed by model)
                             # (384-dim is good balance)

METRIC = "cosine"           # Similarity metric in Pinecone
                             # (Standard for embeddings)
```

---

## How to Rebuild from Scratch

If you want to build this project from the ground up, follow this architecture:

### Phase 1: Project Setup (Hour 1)
```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Create project structure
mkdir Medical-Chatbot
cd Medical-Chatbot

mkdir -p src data templates static logs

# 3. Create files
touch .env requirements.txt setup.py app.py store_index.py
touch src/__init__.py src/helper.py src/prompt.py
touch templates/chat.html static/style.css
```

### Phase 2: Dependencies (Hour 1-2)

Create `requirements.txt`:
```
flask==3.1.1
python-dotenv==1.1.0
langchain==0.3.26
langchain-groq==0.2.0
langchain-pinecone==0.2.8
langchain-community==0.3.26
sentence-transformers==4.1.0
pypdf==5.6.1
pinecone-client==5.0.1
```

Install:
```bash
pip install -r requirements.txt
```

### Phase 3: Core Components (Hour 2-3)

#### 1. Create `src/prompt.py` (Prompt Engineering)
```python
def get_system_prompt_template() -> str:
    return """You are a Medical Assistant. Use provided context to answer.
    Keep responses to 3 sentences max."""
```

#### 2. Create `src/helper.py` (Document Processing)
```python
# Functions for:
# - Loading PDFs from directory
# - Cleaning metadata
# - Chunking documents
# - Initializing embeddings
```

#### 3. Create `store_index.py` (Data Pipeline)
```python
# Orchestrate:
# 1. Load PDFs
# 2. Clean & chunk
# 3. Generate embeddings
# 4. Upload to Pinecone
```

### Phase 4: Backend API (Hour 3-4)

Create `app.py`:
```python
from flask import Flask, render_template, request
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain

app = Flask(__name__)

# Initialize services
embedding_model = initialize_embeddings()
vector_store = PineconeVectorStore.from_existing_index(...)
llm_model = ChatGroq(model="llama-3.1-8b-instant")
rag_pipeline = create_retrieval_chain(retriever, document_chain)

@app.route("/")
def home():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def get_response():
    msg = request.form.get("msg")
    result = rag_pipeline.invoke({"input": msg})
    return result.get("answer", "No response")

app.run(host="0.0.0.0", port=8080)
```

### Phase 5: Frontend (Hour 4-5)

#### Create `templates/chat.html`
```html
<!DOCTYPE html>
<html>
<head>
    <title>Medical Chatbot</title>
    <link rel="stylesheet" href="...">
</head>
<body>
    <div class="chat-container">
        <div id="messages"></div>
        <form id="chatForm">
            <input type="text" name="msg" required>
            <button type="submit">Send</button>
        </form>
    </div>
    <script src="https://code.jquery.com/jquery.min.js"></script>
    <script>
        // Handle form submission
        // Render messages
        // Make AJAX calls
    </script>
</body>
</html>
```

#### Create `static/style.css`
```css
/* Chat interface styling */
/* Responsive design */
/* Modern aesthetics */
```

### Phase 6: Configuration (Hour 5)

Create `.env`:
```
PINECONE_API_KEY=xxx
GROQ_API_KEY=yyy
```

Create `setup.py`:
```python
from setuptools import setup, find_packages
setup(
    name="medical_chatbot",
    version="0.1.0",
    packages=find_packages()
)
```

### Phase 7: Testing & Deployment (Hour 5-6)
```bash
# Add PDFs to data/
cp medical_documents.pdf data/

# Create index
python store_index.py

# Run application
python app.py

# Test: Open http://localhost:8080
```

---

## Database Schema (Vector DB)

### Pinecone Index Structure
```
Index Name: medical-chatbot

Each Record:
{
    "id": "doc_chunk_001",
    "values": [0.12, 0.45, ..., 0.89],  ← 384-dimensional vector
    "metadata": {
        "source": "data/medical_guide.pdf",
        "page": 5,
        "chunk_index": 0
    }
}
```

### Retrieval Example
```
User Query: "What causes diabetes?"

1. Convert to vector: [0.18, 0.22, ..., 0.91]
2. Cosine similarity search in Pinecone
3. Returns:
   [
       {id: "chunk_001", score: 0.95, metadata: {...}},
       {id: "chunk_015", score: 0.89, metadata: {...}},
       {id: "chunk_042", score: 0.82, metadata: {...}}
   ]
```

---

## Performance Optimization Strategies

### 1. **Caching Layer**
```
Query Hash → Cache Check → Response
                ├─ HIT → Return cached
                └─ MISS → Generate & cache
```

### 2. **Batch Processing**
```python
# Instead of processing PDFs one-by-one
for_batch in batch_documents(all_docs, batch_size=10):
    embeddings = model.encode(for_batch)
    pinecone_index.upsert(for_batch)
```

### 3. **Async Processing**
```python
# Frontend
async function sendMessage(msg) {
    let response = await fetch('/get', {method: 'POST', data: msg});
    return response.json();
}

# Backend
from flask import Flask
from threading import Thread

def async_process(query):
    # Process without blocking
    pass
```

### 4. **Vector Quantization**
```
Default: 384-dimensional float32
Optimized: 256-dimensional int8
Tradeoff: Speed vs Accuracy
```

---

## Security by Design

### Input Layer
```
User Input → Validation → Sanitization → Processing
              ├─ Length check
              ├─ Content filtering
              └─ XSS prevention (escapeHtml)
```

### API Layer
```
HTTP Request → Rate Limiting → Authentication → Authorization
               (10/min)       (future)        (future)
```

### Data Layer
```
Sensitive Data:
├─ API Keys → .env file (not in code)
├─ User Queries → Anonymized analytics
└─ Responses → Not stored (stateless)
```

---

## Monitoring & Observability

### Logging Strategy
```
Level 1: ERROR    → API failures, exceptions
Level 2: WARNING  → Rate limit hit, cache miss
Level 3: INFO     → Query received, response sent
Level 4: DEBUG    → Parameter values, internal state
```

### Metrics to Track
```
├─ Query volume (per hour/day)
├─ Response latency (p50, p95, p99)
├─ Cache hit rate (% of queries from cache)
├─ Error rate (% of failed queries)
└─ Cost (API calls to Groq & Pinecone)
```

### Alerting Triggers
```
├─ Response time > 5 seconds
├─ Error rate > 5%
├─ Cache hit rate < 50% (indicates cold start)
└─ API quota usage > 80%
```

---

## Scaling Considerations

### Horizontal Scaling
```
Load Balancer
├── Flask Instance 1 (port 8080)
├── Flask Instance 2 (port 8081)
└── Flask Instance 3 (port 8082)
    ↓
All connect to:
├── Shared Pinecone Index (managed)
└── Shared Redis Cache (optional)
```

### Vertical Scaling
```
Current Bottleneck: Groq API rate limits
Solution: Queue & batch requests
├── Client sends query
├── Added to work queue
├── Worker processes when capacity available
└── Response sent via WebSocket
```

### Database Scaling
```
Pinecone handles:
├─ Automatic replication
├─ Sharding across nodes
├─ 99.99% uptime SLA
└─ Millions of vectors
```

---

## Deployment Options

### Option 1: Local Development
```bash
python app.py
# Application runs on http://localhost:8080
```

### Option 2: Cloud Deployment (Heroku)
```bash
# Procfile
web: python app.py

# Deploy
git push heroku main
```

### Option 3: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

Deploy:
```bash
docker build -t medical-chatbot .
docker run -p 8080:8080 -e PINECONE_API_KEY=xxx medical-chatbot
```

### Option 4: Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-chatbot
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: chatbot
        image: medical-chatbot:latest
        ports:
        - containerPort: 8080
        env:
        - name: PINECONE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: pinecone
```

---

## Essential Technologies Explained

| Tech | Role | Why Chosen |
|------|------|-----------|
| **Flask** | Web framework | Lightweight, perfect for APIs |
| **LangChain** | LLM orchestration | Abstracts RAG complexity |
| **Pinecone** | Vector database | Managed, scales automatically |
| **Groq** | LLM provider | 70B output tokens/min free tier |
| **HuggingFace** | Embeddings | 384-dim, good semantic match |
| **jQuery** | Frontend AJAX | Simple, widely compatible |
| **Bootstrap** | UI Framework | Responsive, professional look |

---

## Conclusion

This medical chatbot demonstrates:
- ✅ Modern RAG architecture
- ✅ Proper separation of concerns
- ✅ Scalable cloud-native design
- ✅ Security-first approach
- ✅ Industry best practices

The code is **production-ready** with minimal enhancements needed for enterprise deployment.

