# Build Medical-Chatbot from Scratch - Step-by-Step Guide

**Estimated Time**: 40-50 hours  
**Difficulty**: Intermediate-Advanced  
**Prerequisites**: Python 3.8+, API keys (Pinecone, Groq)

---

## Phase 1: Project Setup (30 minutes)

### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
```

### Step 2: Create Project Structure
```bash
# Create directories
mkdir Medical-Chatbot
cd Medical-Chatbot

mkdir src data templates static logs

# Create empty files
type nul > .env
type nul > app.py
type nul > store_index.py
type nul > requirements.txt
type nul > setup.py

# Create src files
type nul > src\__init__.py
type nul > src\helper.py
type nul > src\prompt.py

# Create frontend files
type nul > templates\chat.html
type nul > static\style.css

# Create gitignore
type nul > .gitignore
```

### Step 3: Create `.env` file
```env
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
FLASK_ENV=development
FLASK_DEBUG=1
```

**Get API Keys**:
- **Pinecone**: https://www.pinecone.io (free tier: 100K vectors)
- **Groq**: https://console.groq.com (free tier: 70B tokens/month)

---

## Phase 2: Dependencies & Configuration (1 hour)

### Step 4: Create `requirements.txt`
```txt
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

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 6: Create `setup.py`
```python
from setuptools import setup, find_packages

setup(
    name="medical_chatbot",
    version="0.1.0",
    description="RAG-based medical chatbot",
    packages=find_packages(),
    install_requires=[]
)
```

### Step 7: Install Package
```bash
pip install -e .
```

---

## Phase 3: Document Processing Engine (3 hours)

### Step 8: Create `src/prompt.py`
**This file defines how the LLM should behave**

```python
"""
Prompt templates for the medical chatbot.
"""

def get_system_prompt_template() -> str:
    """
    System prompt that guides the LLM's behavior.
    This is the instructions for the AI assistant.
    """
    prompt = (
        "You are a knowledgeable Medical Assistant specialized in answering "
        "health-related questions. Use the provided context from medical "
        "documents to answer user queries accurately. If the information is "
        "not available in the context, politely state that you don't have "
        "that information. Keep responses concise and limited to three "
        "sentences maximum.\n\n"
        "Context:\n{context}"
    )
    return prompt
```

### Step 9: Create `src/helper.py`
**This file handles PDF loading, chunking, and embedding**

```python
"""
Helper functions for document processing.
This is the data pipeline.
"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List


def load_pdf_documents(directory_path: str) -> List[Document]:
    """
    Load all PDF files from directory.
    
    This reads medical PDFs and converts them to Document objects.
    """
    loader = DirectoryLoader(
        directory_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} pages from PDFs")
    return documents


def clean_document_metadata(documents: List[Document]) -> List[Document]:
    """
    Remove unnecessary metadata.
    
    Keep only source filename to reduce data size.
    """
    cleaned_docs = []
    
    for doc in documents:
        source_path = doc.metadata.get("source", "")
        
        cleaned_doc = Document(
            page_content=doc.page_content,
            metadata={"source": source_path}
        )
        cleaned_docs.append(cleaned_doc)
    
    return cleaned_docs


def split_documents_into_chunks(documents: List[Document]) -> List[Document]:
    """
    Split large documents into smaller chunks.
    
    Why chunks?
    - Better semantic relevance in retrieval
    - Fits within context windows
    - Improves search accuracy
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Characters per chunk
        chunk_overlap=20     # Overlap between chunks
    )
    
    text_chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(text_chunks)} chunks")
    return text_chunks


def initialize_embeddings():
    """
    Load HuggingFace embeddings model.
    
    Model: all-MiniLM-L6-v2
    - Output: 384-dimensional vectors
    - Speed: Fast inference
    - Quality: Good semantic understanding
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    print("✓ Embeddings model loaded")
    return embeddings
```

### Step 10: Create `src/__init__.py`
```python
"""Medical Chatbot Package"""

__version__ = "0.1.0"
```

---

## Phase 4: Data Pipeline (2 hours)

### Step 11: Create `store_index.py`
**This converts PDFs → chunks → embeddings → Pinecone**

```python
"""
Pipeline to create Pinecone index from PDF documents.
Run: python store_index.py
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf_documents,
    clean_document_metadata,
    split_documents_into_chunks,
    initialize_embeddings
)

load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DATA_DIRECTORY = "data/"
INDEX_NAME = "medical-chatbot"
EMBEDDING_DIMENSION = 384


def create_pinecone_index(pc_client: Pinecone, index_name: str) -> None:
    """Create Pinecone index if it doesn't exist."""
    
    if pc_client.has_index(index_name):
        print(f"✓ Index '{index_name}' already exists")
        return
    
    print(f"Creating index '{index_name}'...")
    
    pc_client.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    print(f"✓ Index created!")


def main():
    """Main pipeline execution."""
    
    print("\n" + "="*60)
    print("Medical Chatbot - Index Creation Pipeline")
    print("="*60 + "\n")
    
    # Step 1: Load PDFs
    print("Step 1: Loading PDFs from data/...")
    raw_documents = load_pdf_documents(DATA_DIRECTORY)
    
    if not raw_documents:
        print("❌ No PDFs found! Add medical PDFs to data/ directory")
        return
    
    # Step 2: Clean metadata
    print("\nStep 2: Cleaning metadata...")
    cleaned_documents = clean_document_metadata(raw_documents)
    
    # Step 3: Split chunks
    print("\nStep 3: Splitting into chunks...")
    chunks = split_documents_into_chunks(cleaned_documents)
    
    # Step 4: Initialize embeddings
    print("\nStep 4: Loading embeddings model...")
    embeddings_model = initialize_embeddings()
    
    # Step 5: Connect to Pinecone
    print("\nStep 5: Connecting to Pinecone...")
    pc_client = Pinecone(api_key=PINECONE_API_KEY)
    
    # Step 6: Create index
    print("\nStep 6: Creating/verifying index...")
    create_pinecone_index(pc_client, INDEX_NAME)
    
    # Step 7: Upload documents
    print("\nStep 7: Uploading documents to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=INDEX_NAME,
        embedding=embeddings_model
    )
    
    print("\n" + "="*60)
    print("✓ Pipeline completed!")
    print("="*60)
    print("\nNext: python app.py\n")


if __name__ == "__main__":
    main()
```

### Step 12: Prepare Sample Data
```bash
# Create data directory if not exists
mkdir data

# Add medical PDF files here
# You can use sample PDFs or real medical documents
```

---

## Phase 5: Flask Backend (4 hours)

### Step 13: Create `app.py` - Main Web Server
**This is the core API and web server**

```python
"""
Medical Chatbot - Flask Application
Main web server handling chat interface and API endpoints.
"""

import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import initialize_embeddings
from src.prompt import get_system_prompt_template

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = "medical-chatbot"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# Set environment variables for LangChain
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

print("Initializing Medical Chatbot...")

# Initialize embeddings
embedding_model = initialize_embeddings()

# Connect to Pinecone
vector_store = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model
)

# Create retriever
document_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize LLM
llm_model = ChatGroq(
    model=GROQ_MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0
)

# Create RAG chain
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt_template()),
    ("human", "{input}"),
])

document_chain = create_stuff_documents_chain(llm_model, chat_prompt)
rag_pipeline = create_retrieval_chain(document_retriever, document_chain)

print("✓ Chatbot initialized!\n")


# Routes
@app.route("/")
def home():
    """Render chat interface."""
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def get_response():
    """Handle chat requests."""
    user_message = request.form.get("msg", "").strip()
    
    if not user_message:
        return "Please provide a valid question."
    
    try:
        result = rag_pipeline.invoke({"input": user_message})
        bot_response = result.get("answer", "I'm sorry, I couldn't generate a response.")
        return str(bot_response)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return "An error occurred. Please try again."


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
```

---

## Phase 6: Frontend (6 hours)

### Step 14: Create `templates/chat.html`
**This is the user-facing chat interface**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Chatbot</title>
    
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css">
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css">
    
    <!-- Custom CSS -->
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container-fluid h-100">
        <div class="row justify-content-center h-100">
            <div class="col-md-8 col-xl-6 chat">
                <div class="card">
                    
                    <!-- Header -->
                    <div class="card-header msg_head">
                        <div class="d-flex bd-highlight">
                            <div class="img_cont">
                                <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" 
                                     class="rounded-circle user_img" alt="Bot">
                                <span class="online_icon"></span>
                            </div>
                            <div class="user_info">
                                <span>Medical Chatbot</span>
                                <p>Ask me medical questions!</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Messages -->
                    <div id="chatMessages" class="card-body msg_card_body"></div>
                    
                    <!-- Input -->
                    <div class="card-footer">
                        <form id="chatForm" class="input-group">
                            <input type="text" id="userInput" name="msg" 
                                   placeholder="Type your question..." required
                                   class="form-control type_msg">
                            <div class="input-group-append">
                                <button type="submit" class="input-group-text send_btn">
                                    <i class="fas fa-location-arrow"></i>
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- jQuery -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
    
    <!-- Bootstrap JS -->
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js"></script>
    
    <!-- Chat Logic -->
    <script>
        $(document).ready(function() {
            const chatForm = $("#chatForm");
            const chatMessages = $("#chatMessages");
            const userInput = $("#userInput");
            
            function escapeHtml(text) {
                const map = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'};
                return text.replace(/[&<>"']/g, m => map[m]);
            }
            
            function getTime() {
                const now = new Date();
                return now.getHours() + ":" + String(now.getMinutes()).padStart(2, '0');
            }
            
            chatForm.on("submit", function(e) {
                e.preventDefault();
                
                const msg = userInput.val().trim();
                if (!msg) return;
                
                const time = getTime();
                
                // Display user message
                chatMessages.append(`
                    <div class="d-flex justify-content-end mb-4">
                        <div class="msg_cotainer_send">
                            ${escapeHtml(msg)}
                            <span class="msg_time_send">${time}</span>
                        </div>
                        <div class="img_cont_msg">
                            <img src="https://i.ibb.co/d5b84Xw/Untitled-design.png" 
                                 class="rounded-circle user_img_msg" alt="You">
                        </div>
                    </div>
                `);
                
                userInput.val("");
                chatForm.find("input, button").prop("disabled", true);
                
                // Typing indicator
                chatMessages.append(`
                    <div class="d-flex justify-content-start mb-4" id="typing">
                        <div class="img_cont_msg">
                            <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" 
                                 class="rounded-circle user_img_msg" alt="Bot">
                        </div>
                        <div class="msg_cotainer">Typing...</div>
                    </div>
                `);
                
                chatMessages.scrollTop(chatMessages[0].scrollHeight);
                
                // Send request
                $.ajax({
                    url: "/get",
                    type: "POST",
                    data: { msg: msg },
                    success: function(response) {
                        $("#typing").remove();
                        chatMessages.append(`
                            <div class="d-flex justify-content-start mb-4">
                                <div class="img_cont_msg">
                                    <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" 
                                         class="rounded-circle user_img_msg" alt="Bot">
                                </div>
                                <div class="msg_cotainer">
                                    ${escapeHtml(response)}
                                    <span class="msg_time">${time}</span>
                                </div>
                            </div>
                        `);
                        chatMessages.scrollTop(chatMessages[0].scrollHeight);
                    },
                    error: function() {
                        $("#typing").remove();
                        chatMessages.append(`
                            <div class="d-flex justify-content-start mb-4">
                                <div class="img_cont_msg">
                                    <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" 
                                         class="rounded-circle user_img_msg" alt="Bot">
                                </div>
                                <div class="msg_cotainer">
                                    Error: Could not get response
                                    <span class="msg_time">${time}</span>
                                </div>
                            </div>
                        `);
                        chatMessages.scrollTop(chatMessages[0].scrollHeight);
                    },
                    complete: function() {
                        chatForm.find("input, button").prop("disabled", false);
                        userInput.focus();
                    }
                });
            });
            
            userInput.focus();
        });
    </script>
</body>
</html>
```

### Step 15: Create `static/style.css`
**This styles the chat interface**

```css
/* Global */
body, html {
    height: 100%;
    margin: 0;
    background: linear-gradient(to right, #26334a, #323c46, #212147);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Chat */
.chat {
    margin: auto;
}

.card {
    height: 500px;
    border-radius: 15px;
    background-color: rgba(0, 0, 0, 0.4);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    display: flex;
    flex-direction: column;
}

/* Messages */
.msg_card_body {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
}

.msg_card_body::-webkit-scrollbar {
    width: 6px;
}

.msg_card_body::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 10px;
}

/* Header */
.card-header {
    border-radius: 15px 15px 0 0;
    border-bottom: 0;
    background-color: rgba(0, 0, 0, 0.3);
}

.card-footer {
    border-radius: 0 0 15px 15px;
    border-top: 0;
    background-color: rgba(0, 0, 0, 0.3);
}

/* User Info */
.user_info {
    margin: auto 0 auto 15px;
}

.user_info span {
    font-size: 20px;
    color: white;
    font-weight: 500;
}

.user_info p {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.6);
    margin: 0;
}

/* Images */
.user_img {
    height: 70px;
    width: 70px;
    border-radius: 50%;
    object-fit: cover;
}

.user_img_msg {
    height: 40px;
    width: 40px;
    border-radius: 50%;
    object-fit: cover;
}

.img_cont {
    height: 70px;
    width: 70px;
    position: relative;
}

.img_cont_msg {
    height: 40px;
    width: 40px;
}

.online_icon {
    position: absolute;
    height: 15px;
    width: 15px;
    background-color: #4cd137;
    border-radius: 50%;
    bottom: 0.2em;
    right: 0.4em;
    border: 1.5px solid white;
}

/* Input */
.type_msg {
    background-color: rgba(0, 0, 0, 0.3);
    border: 0;
    color: white;
    height: 60px;
}

.type_msg:focus {
    box-shadow: none;
    background-color: rgba(0, 0, 0, 0.4);
    color: white;
}

.type_msg::placeholder {
    color: rgba(255, 255, 255, 0.5);
}

/* Send Button */
.send_btn {
    border-radius: 0 15px 15px 0;
    background-color: rgba(0, 0, 0, 0.3);
    border: 0;
    color: white;
    cursor: pointer;
    transition: background-color 0.3s;
}

.send_btn:hover {
    background-color: rgba(0, 0, 0, 0.5);
}

.send_btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Messages */
.msg_cotainer {
    margin: auto 10px;
    border-radius: 25px;
    background-color: rgb(82, 172, 255);
    padding: 10px 15px;
    position: relative;
    max-width: 70%;
    word-wrap: break-word;
    color: white;
}

.msg_cotainer_send {
    margin: auto 10px;
    border-radius: 25px;
    background-color: #58cc71;
    padding: 10px 15px;
    position: relative;
    max-width: 70%;
    word-wrap: break-word;
    color: white;
}

/* Timestamps */
.msg_time {
    position: absolute;
    left: 0;
    bottom: -15px;
    color: rgba(255, 255, 255, 0.5);
    font-size: 10px;
}

.msg_time_send {
    position: absolute;
    right: 0;
    bottom: -15px;
    color: rgba(255, 255, 255, 0.5);
    font-size: 10px;
}

/* Responsive */
@media (max-width: 576px) {
    .card {
        height: 100vh;
        border-radius: 0;
    }
}
```

---

## Phase 7: Test & Run (1 hour)

### Step 16: Add Sample Medical PDFs
```bash
# Create sample PDF directory
mkdir data

# Add your medical PDF files here
# Examples:
# - Medical handbook
# - Disease information documents
# - Health guidelines
```

### Step 17: Create Vector Index
```bash
# This converts PDFs into embeddings in Pinecone
python store_index.py

# Expected output:
# ✓ Loaded X pages
# ✓ Created Y chunks
# ✓ Index created/verified
# ✓ Documents uploaded
```

### Step 18: Start the Application
```bash
# Run the Flask server
python app.py

# Output:
# Initializing Medical Chatbot...
# ✓ Chatbot initialized!
# * Running on http://0.0.0.0:8080
```

### Step 19: Test in Browser
```
Open: http://localhost:8080
Type a medical question
Get AI-powered response!
```

---

## Phase 8: Security & Production (varies)

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for:
- Rate limiting
- Input validation
- Error handling
- Logging
- Security headers
- Caching
- Analytics

---

## Complete Checklist

- [ ] Phase 1: Project Setup (30 min)
- [ ] Phase 2: Dependencies (1 hour)
- [ ] Phase 3: Document Processing (3 hours)
- [ ] Phase 4: Data Pipeline (2 hours)
- [ ] Phase 5: Flask Backend (4 hours)
- [ ] Phase 6: Frontend (6 hours)
- [ ] Phase 7: Test & Run (1 hour)
- [ ] Phase 8: Production Ready (varies)

**Total: 19-22 hours of hands-on work**

---

## Key Commands Reference

```bash
# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Create index
python store_index.py

# Run application
python app.py

# Access
http://localhost:8080
```

---

## Common Issues & Fixes

**Issue**: "No module named 'src'"  
**Fix**: Ensure you're in the Medical-Chatbot directory and ran `pip install -e .`

**Issue**: "Pinecone connection failed"  
**Fix**: Check your `PINECONE_API_KEY` in `.env` is correct

**Issue**: "No PDFs found"  
**Fix**: Add medical PDF files to the `data/` directory first

**Issue**: "Groq API error"  
**Fix**: Check your `GROQ_API_KEY` in `.env` is valid

---

**Start with Phase 1 and work through systematically. Each phase builds on the previous!**

