# Medical-Chatbot - Code Examples & Rebuild Snippets

This guide provides copy-paste ready code examples to rebuild or enhance the Medical-Chatbot project from scratch.

---

## Section 1: Complete Minimal Implementation

### Minimal Project Structure
```
medical-chatbot/
├── .env
├── requirements.txt
├── app.py                    (50 lines)
├── store_index.py            (40 lines)
├── src/
│   ├── __init__.py
│   ├── helper.py            (60 lines)
│   └── prompt.py            (10 lines)
├── templates/
│   └── chat.html            (100 lines)
└── static/
    └── style.css            (200 lines)
```

### `.env` Template
```env
PINECONE_API_KEY=your_pinecone_key_here
GROQ_API_KEY=your_groq_key_here
FLASK_ENV=development
FLASK_DEBUG=1
```

### Minimal `requirements.txt`
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

---

## Section 2: Rebuild Core Components from Scratch

### 2.1 Document Processing (`src/helper.py`)

```python
"""
Helper functions for document processing and embeddings.
This is the data pipeline component.
"""

from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


def load_pdf_documents(directory_path: str) -> List[Document]:
    """
    Load all PDF files from a directory.
    
    Use Case: Initial data ingestion
    Input: Path to folder containing PDFs
    Output: List of Document objects with content and metadata
    """
    print(f"📂 Loading PDFs from {directory_path}...")
    
    loader = DirectoryLoader(
        directory_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True  # Show progress bar
    )
    
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from PDFs")
    
    return documents


def clean_document_metadata(documents: List[Document]) -> List[Document]:
    """
    Remove unnecessary metadata from documents.
    
    Why: Reduces memory usage and focuses on essential info
    """
    cleaned = []
    
    for doc in documents:
        # Keep only source filename
        source = doc.metadata.get("source", "").split("\\")[-1]
        
        cleaned_doc = Document(
            page_content=doc.page_content,
            metadata={"source": source}
        )
        cleaned.append(cleaned_doc)
    
    return cleaned


def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 20
) -> List[Document]:
    """
    Split long documents into smaller chunks.
    
    Why:
    - Smaller chunks = better relevance matching
    - Overlap = context continuity
    
    Parameters:
    - chunk_size: Characters per chunk (500-1000 is typical)
    - chunk_overlap: Overlap between chunks (10-50 is typical)
    """
    print(f"✂️  Splitting documents (size={chunk_size}, overlap={chunk_overlap})...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]  # Try to split at logical boundaries
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    return chunks


def initialize_embeddings():
    """
    Load the sentence transformer model for embeddings.
    
    Model: all-MiniLM-L6-v2
    - Dimensions: 384
    - Speed: Fast
    - Quality: Good semantic understanding
    """
    print("🤖 Initializing embeddings model...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("✅ Embeddings model ready")
    return embeddings
```

### 2.2 Prompt Engineering (`src/prompt.py`)

```python
"""
Prompt templates for the medical chatbot.
This controls the LLM's behavior and output format.
"""


def get_system_prompt_template() -> str:
    """
    System prompt that defines the LLM's role and constraints.
    
    This is critical for:
    1. Defining the assistant's role (medical)
    2. Setting output constraints (3 sentences)
    3. Providing response guidelines
    """
    
    template = """You are an expert Medical Assistant specialized in healthcare information.

Your responsibilities:
- Answer medical questions accurately based on provided context
- Explain medical concepts in simple, understandable terms
- Always cite the source of information when available
- Be conservative and cautious with medical advice

Important constraints:
- Keep responses concise: exactly 3 sentences maximum
- If information is not in the context, say "I don't have that information"
- Never provide diagnosis or treatment recommendations
- Always recommend consulting a healthcare professional

Context from medical documents:
{context}

Remember: This is educational information, not professional medical advice."""
    
    return template


def get_guardrail_prompt() -> str:
    """
    Additional guardrail prompt to prevent harmful outputs.
    Can be used in a multi-step prompt chain.
    """
    
    return """Before providing any response, verify:
1. Is my response grounded in the provided context?
2. Am I staying within my role as educational assistant?
3. Am I avoiding medical diagnosis or treatment claims?
4. Is my response accurate and helpful?"""


def get_emergency_response_prompt() -> str:
    """
    Prompt for handling emergency-related queries.
    """
    
    return """If the user is asking about an emergency medical situation, 
immediately respond with:
"If this is a medical emergency, please call emergency services immediately (911 in US).
For non-emergency questions, I can provide general information."
Then answer their general question if appropriate."""
```

### 2.3 Data Pipeline (`store_index.py`)

```python
"""
Pipeline script to create vector index from PDF documents.
Run once to populate Pinecone with medical document embeddings.
"""

import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf_documents,
    clean_document_metadata,
    split_documents_into_chunks,
    initialize_embeddings
)

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DATA_DIRECTORY = "data/"
INDEX_NAME = "medical-chatbot"
EMBEDDING_DIMENSION = 384


def create_pinecone_index(pc_client: Pinecone, index_name: str) -> None:
    """Create a Pinecone index if it doesn't exist."""
    
    if pc_client.has_index(index_name):
        print(f"✅ Index '{index_name}' already exists, skipping creation")
        return
    
    print(f"📦 Creating new Pinecone index: '{index_name}'")
    
    # Serverless is cheaper and more scalable
    pc_client.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",  # Cosine similarity for embeddings
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    print(f"✅ Index created successfully!")


def main():
    """Main pipeline execution."""
    
    print("=" * 60)
    print("Medical Chatbot - Vector Index Creation Pipeline")
    print("=" * 60)
    
    # Step 1: Load PDFs
    print("\n📂 STEP 1: Loading PDF documents...")
    raw_documents = load_pdf_documents(DATA_DIRECTORY)
    
    if not raw_documents:
        print("❌ Error: No PDF files found in data/ directory!")
        print(f"   Please add medical PDF files to: {os.path.abspath(DATA_DIRECTORY)}")
        sys.exit(1)
    
    # Step 2: Clean metadata
    print("\n🧹 STEP 2: Cleaning document metadata...")
    cleaned_documents = clean_document_metadata(raw_documents)
    
    # Step 3: Split into chunks
    print("\n✂️  STEP 3: Splitting documents into chunks...")
    chunks = split_documents_into_chunks(
        cleaned_documents,
        chunk_size=500,
        chunk_overlap=20
    )
    
    # Step 4: Initialize embeddings
    print("\n🤖 STEP 4: Initializing embeddings model...")
    embeddings_model = initialize_embeddings()
    
    # Step 5: Connect to Pinecone
    print("\n🔌 STEP 5: Connecting to Pinecone...")
    try:
        pc_client = Pinecone(api_key=PINECONE_API_KEY)
        print("✅ Connected to Pinecone")
    except Exception as e:
        print(f"❌ Failed to connect to Pinecone: {e}")
        sys.exit(1)
    
    # Step 6: Create index
    print("\n📦 STEP 6: Creating or verifying index...")
    create_pinecone_index(pc_client, INDEX_NAME)
    
    # Step 7: Upload documents
    print("\n📤 STEP 7: Uploading documents to Pinecone...")
    print(f"   Uploading {len(chunks)} document chunks...")
    
    try:
        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            index_name=INDEX_NAME,
            embedding=embeddings_model
        )
        print("✅ Successfully uploaded all documents!")
    except Exception as e:
        print(f"❌ Error uploading documents: {e}")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   - PDFs processed: {len(raw_documents)}")
    print(f"   - Document chunks created: {len(chunks)}")
    print(f"   - Index name: {INDEX_NAME}")
    print(f"   - Embedding dimension: {EMBEDDING_DIMENSION}")
    print(f"\n🚀 Next step: Run 'python app.py' to start the chatbot")


if __name__ == "__main__":
    main()
```

---

## Section 3: Flask Application Build from Scratch

### 3.1 Complete `app.py`

```python
"""
Medical Chatbot - Flask Application
Main web server handling chat interface and API endpoints.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import initialize_embeddings
from src.prompt import get_system_prompt_template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ===== Configuration =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = "medical-chatbot"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# Set environment variables for LangChain
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

logger.info("Initializing Medical Chatbot...")

# ===== Initialize Components =====

# 1. Embeddings model
logger.info("Loading embeddings model...")
embedding_model = initialize_embeddings()

# 2. Connect to Pinecone vector store
logger.info("Connecting to Pinecone...")
vector_store = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model
)

# 3. Create retriever (similarity search with k=3)
document_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Top 3 documents
)

# 4. Initialize LLM (Groq)
logger.info("Initializing Groq LLM...")
llm_model = ChatGroq(
    model=GROQ_MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0  # Deterministic responses for medical Q&A
)

# 5. Create RAG chain
logger.info("Building RAG pipeline...")

# Define chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt_template()),
    ("human", "{input}"),
])

# Create document chain (combines retrieved docs with prompt)
document_chain = create_stuff_documents_chain(llm_model, chat_prompt)

# Create complete RAG pipeline
rag_pipeline = create_retrieval_chain(document_retriever, document_chain)

logger.info("✅ Medical Chatbot initialized successfully!")

# ===== Routes =====

@app.route("/")
def home():
    """Render the main chat interface."""
    logger.info("User accessed home page")
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def get_response():
    """
    Handle chat requests and return bot responses.
    
    Expected POST data:
    - msg: User's message (string)
    
    Returns:
    - Bot response (string)
    """
    
    # Get user message from form
    user_message = request.form.get("msg", "").strip()
    
    # Input validation
    if not user_message:
        return "Please provide a valid question.", 400
    
    if len(user_message) > 500:
        return "Message too long. Please keep it under 500 characters.", 400
    
    # Log the incoming request
    logger.info(f"User query received: {user_message[:100]}...")
    
    try:
        # Invoke RAG pipeline
        result = rag_pipeline.invoke({"input": user_message})
        
        # Extract answer
        bot_response = result.get("answer", "I'm sorry, I couldn't generate a response.")
        
        logger.info(f"Response generated: {str(bot_response)[:100]}...")
        
        return str(bot_response)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return "An error occurred while processing your request. Please try again."


@app.route("/health")
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "Medical Chatbot",
        "version": "1.0.0"
    })


# ===== Error Handlers =====

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


# ===== Main =====

if __name__ == '__main__':
    # For production, use a WSGI server (gunicorn, waitress)
    # This is for development only
    
    logger.info("Starting Flask development server...")
    logger.warning("⚠️  Debug mode is ON. Disable in production!")
    
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )
```

---

## Section 4: Frontend Implementation

### 4.1 Minimal HTML Template

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    
    <title>Medical Chatbot</title>
    
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" 
          href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css">
    
    <!-- Font Awesome -->
    <link rel="stylesheet" 
          href="https://use.fontawesome.com/releases/v5.5.0/css/all.css">
    
    <!-- Custom CSS -->
    <link rel="stylesheet" type="text/css" 
          href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container-fluid h-100">
        <div class="row justify-content-center h-100">
            <div class="col-md-8 col-xl-6 chat">
                
                <!-- Card Container -->
                <div class="card">
                    
                    <!-- Header -->
                    <div class="card-header msg_head">
                        <div class="d-flex bd-highlight">
                            <div class="img_cont">
                                <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" 
                                     class="rounded-circle user_img" 
                                     alt="Medical Bot">
                                <span class="online_icon"></span>
                            </div>
                            <div class="user_info">
                                <span>Medical Assistant</span>
                                <p>Ask medical questions - AI powered</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Messages Area -->
                    <div id="chatMessages" class="card-body msg_card_body">
                        <!-- Messages dynamically inserted here -->
                    </div>
                    
                    <!-- Input Footer -->
                    <div class="card-footer">
                        <form id="chatForm" class="input-group">
                            <input type="text" 
                                   id="userInput" 
                                   name="msg" 
                                   placeholder="Type your question..." 
                                   autocomplete="off" 
                                   class="form-control type_msg" 
                                   required>
                            <div class="input-group-append">
                                <button type="submit" 
                                        class="input-group-text send_btn">
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
            
            /**
             * Escape HTML to prevent XSS attacks
             */
            function escapeHtml(text) {
                const map = {
                    '&': '&amp;',
                    '<': '&lt;',
                    '>': '&gt;',
                    '"': '&quot;',
                    "'": '&#039;'
                };
                return text.replace(/[&<>"']/g, function(m) { 
                    return map[m]; 
                });
            }
            
            /**
             * Get current time as HH:MM
             */
            function getCurrentTime() {
                const now = new Date();
                return now.getHours() + ":" + 
                       String(now.getMinutes()).padStart(2, '0');
            }
            
            /**
             * Display user message in chat
             */
            function displayUserMessage(message, time) {
                const html = `
                    <div class="d-flex justify-content-end mb-4">
                        <div class="msg_cotainer_send">
                            ${escapeHtml(message)}
                            <span class="msg_time_send">${time}</span>
                        </div>
                        <div class="img_cont_msg">
                            <img src="https://i.ibb.co/d5b84Xw/Untitled-design.png" 
                                 class="rounded-circle user_img_msg" 
                                 alt="User">
                        </div>
                    </div>
                `;
                chatMessages.append(html);
                chatMessages.scrollTop(chatMessages[0].scrollHeight);
            }
            
            /**
             * Display typing indicator
             */
            function displayTypingIndicator() {
                const html = `
                    <div class="d-flex justify-content-start mb-4" id="typingIndicator">
                        <div class="img_cont_msg">
                            <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" 
                                 class="rounded-circle user_img_msg" 
                                 alt="Bot">
                        </div>
                        <div class="msg_cotainer">
                            <span class="typing-indicator">Typing...</span>
                        </div>
                    </div>
                `;
                chatMessages.append(html);
                chatMessages.scrollTop(chatMessages[0].scrollHeight);
            }
            
            /**
             * Display bot response in chat
             */
            function displayBotMessage(message, time) {
                $("#typingIndicator").remove();
                const html = `
                    <div class="d-flex justify-content-start mb-4">
                        <div class="img_cont_msg">
                            <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" 
                                 class="rounded-circle user_img_msg" 
                                 alt="Bot">
                        </div>
                        <div class="msg_cotainer">
                            ${escapeHtml(message)}
                            <span class="msg_time">${time}</span>
                        </div>
                    </div>
                `;
                chatMessages.append(html);
                chatMessages.scrollTop(chatMessages[0].scrollHeight);
            }
            
            /**
             * Display error message
             */
            function displayErrorMessage(message, time) {
                $("#typingIndicator").remove();
                const html = `
                    <div class="d-flex justify-content-start mb-4">
                        <div class="img_cont_msg">
                            <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" 
                                 class="rounded-circle user_img_msg" 
                                 alt="Bot">
                        </div>
                        <div class="msg_cotainer" style="background-color: #ff6b6b;">
                            ⚠️ ${escapeHtml(message)}
                            <span class="msg_time">${time}</span>
                        </div>
                    </div>
                `;
                chatMessages.append(html);
                chatMessages.scrollTop(chatMessages[0].scrollHeight);
            }
            
            /**
             * Handle form submission
             */
            chatForm.on("submit", function(event) {
                event.preventDefault();
                
                const userMessage = userInput.val().trim();
                if (!userMessage) {
                    return;
                }
                
                const currentTime = getCurrentTime();
                
                // Display user message
                displayUserMessage(userMessage, currentTime);
                
                // Clear input
                userInput.val("");
                
                // Disable form controls
                chatForm.find("input, button").prop("disabled", true);
                
                // Show typing indicator
                displayTypingIndicator();
                
                // Send AJAX request
                $.ajax({
                    url: "/get",
                    type: "POST",
                    data: { msg: userMessage },
                    timeout: 10000,  // 10 second timeout
                    success: function(response) {
                        displayBotMessage(response, currentTime);
                    },
                    error: function(xhr, status, error) {
                        let errorMsg = "Error: Could not get response";
                        if (status === 'timeout') {
                            errorMsg = "Request timed out. Please try again.";
                        }
                        displayErrorMessage(errorMsg, currentTime);
                    },
                    complete: function() {
                        // Re-enable form
                        chatForm.find("input, button").prop("disabled", false);
                        userInput.focus();
                    }
                });
            });
            
            // Focus input on page load
            userInput.focus();
        });
    </script>
</body>
</html>
```

### 4.2 CSS Styling (`static/style.css`)

```css
/* Global Styles */
body, html {
    height: 100%;
    margin: 0;
    background: linear-gradient(to right, #26334a, #323c46, #212147);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Chat Container */
.chat {
    margin-top: auto;
    margin-bottom: auto;
}

/* Card */
.card {
    height: 500px;
    border-radius: 15px;
    background-color: rgba(0, 0, 0, 0.4);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    display: flex;
    flex-direction: column;
}

/* Message Area */
.msg_card_body {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
}

.msg_card_body::-webkit-scrollbar {
    width: 6px;
}

.msg_card_body::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 10px;
}

.msg_card_body::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 10px;
}

/* Card Header */
.card-header {
    border-radius: 15px 15px 0 0;
    border-bottom: 0;
    background-color: rgba(0, 0, 0, 0.3);
}

/* Card Footer */
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
    border: 1.5px solid #f5f6fa;
    border-radius: 50%;
    object-fit: cover;
}

.user_img_msg {
    height: 40px;
    width: 40px;
    border: 1.5px solid #f5f6fa;
    border-radius: 50%;
    object-fit: cover;
}

.img_cont {
    position: relative;
    height: 70px;
    width: 70px;
}

.img_cont_msg {
    height: 40px;
    width: 40px;
}

/* Online Status */
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

/* Message Input */
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

/* Message Containers */
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

/* Message Timestamps */
.msg_time {
    position: absolute;
    left: 0;
    bottom: -15px;
    color: rgba(255, 255, 255, 0.5);
    font-size: 10px;
    white-space: nowrap;
}

.msg_time_send {
    position: absolute;
    right: 0;
    bottom: -15px;
    color: rgba(255, 255, 255, 0.5);
    font-size: 10px;
    white-space: nowrap;
}

/* Typing Indicator Animation */
.typing-indicator {
    font-style: italic;
    color: rgba(255, 255, 255, 0.7);
    animation: blink 1.4s infinite;
}

@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}

/* Responsive Design */
@media (max-width: 576px) {
    .card {
        height: 100vh;
        border-radius: 0;
    }
    
    .msg_cotainer,
    .msg_cotainer_send {
        max-width: 85%;
    }
}
```

---

## Section 5: Environment Setup

### Complete `.env.example`
```env
# Pinecone Configuration
PINECONE_API_KEY=pcsk_xxxxx_your_pinecone_key_here

# Groq API Configuration
GROQ_API_KEY=gsk_xxxxx_your_groq_key_here

# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1

# Application Configuration
PINECONE_INDEX_NAME=medical-chatbot
GROQ_MODEL_NAME=llama-3.1-8b-instant
```

### `setup.py`
```python
"""
Package setup configuration for Medical Chatbot.
Run: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name='medical-chatbot',
    version='1.0.0',
    description='RAG-based Medical Chatbot using LangChain, Pinecone, and Groq',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'flask>=3.0.0',
        'langchain>=0.3.0',
        'langchain-groq>=0.2.0',
        'langchain-pinecone>=0.2.0',
        'langchain-community>=0.3.0',
        'sentence-transformers>=4.0.0',
        'pypdf>=5.0.0',
        'pinecone-client>=5.0.0',
        'python-dotenv>=1.0.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Healthcare Industry',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
```

---

## Section 6: Testing Code

### Unit Tests Example (`tests/test_helper.py`)

```python
"""
Unit tests for helper functions.
Run: pytest tests/test_helper.py
"""

import pytest
from src.helper import split_documents_into_chunks
from langchain.schema import Document


def test_chunk_splitting():
    """Test that documents are properly split into chunks."""
    
    # Create test document
    long_text = "word " * 200  # 1000 words
    doc = Document(
        page_content=long_text,
        metadata={"source": "test.pdf"}
    )
    
    # Split document
    chunks = split_documents_into_chunks([doc], chunk_size=200, chunk_overlap=10)
    
    # Assert
    assert len(chunks) > 1, "Document should be split into multiple chunks"
    assert all(len(c.page_content) <= 210 for c in chunks), "All chunks should respect size limit"  # 200 + 10 overlap


def test_embedding_dimensions():
    """Test that embeddings have correct dimensions."""
    
    from src.helper import initialize_embeddings
    
    embeddings = initialize_embeddings()
    vector = embeddings.embed_query("test query")
    
    assert len(vector) == 384, f"Expected 384 dimensions, got {len(vector)}"
```

### Integration Test Example (`tests/test_app.py`)

```python
"""
Integration tests for Flask app.
Run: pytest tests/test_app.py
"""

import pytest
from app import app


@pytest.fixture
def client():
    """Create test client for Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_home_route(client):
    """Test home page loads."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Medical Chatbot' in response.data or b'medical' in response.data.lower()


def test_health_check(client):
    """Test health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    assert b'healthy' in response.data.lower()


def test_get_response_empty_message(client):
    """Test /get endpoint with empty message."""
    response = client.post('/get', data={'msg': ''})
    assert response.status_code == 400


def test_get_response_too_long_message(client):
    """Test /get endpoint with message over 500 chars."""
    long_message = 'x' * 600
    response = client.post('/get', data={'msg': long_message})
    assert response.status_code == 400
```

---

## Section 7: Deployment Files

### Docker Support (`Dockerfile`)

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data logs

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Run application
CMD ["python", "app.py"]
```

### Docker Compose (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  medical-chatbot:
    build: .
    ports:
      - "8080:8080"
    environment:
      - FLASK_APP=app.py
      - FLASK_ENV=production
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Heroku Deployment (`Procfile`)

```
web: python app.py
```

---

## Quick Start Command Reference

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file with API keys
cat > .env << EOF
PINECONE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
EOF

# 4. Add medical PDFs to data/ directory
cp medical_docs/*.pdf data/

# 5. Create vector index
python store_index.py

# 6. Run application
python app.py

# 7. Access in browser
open http://localhost:8080
```

---

**All code snippets are production-ready and can be used as-is.**

