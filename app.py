"""
Medical Chatbot Application
A Flask-based RAG chatbot for medical queries using LangChain, Pinecone, and Groq
"""

import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone as PineconeVectorStore
import pinecone
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import initialize_embeddings
from src.prompt import get_system_prompt_template

# Load environment variables
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = "medical-chatbot"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize embeddings
embedding_model = initialize_embeddings()

# Connect to Pinecone vector store
vector_store = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model
)

# Create retriever
document_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize Groq chat model
llm_model = ChatGroq(
    model=GROQ_MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0
)

# Create prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", get_system_prompt_template()),
    ("human", "{input}"),
])

# Build RAG chain
document_chain = create_stuff_documents_chain(llm_model, chat_prompt)
rag_pipeline = create_retrieval_chain(document_retriever, document_chain)


@app.route("/")
def home():
    """Render the main chat interface"""
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def get_response():
    """Handle chat requests and return bot responses"""
    user_message = request.form.get("msg", "")
    
    if not user_message.strip():
        return "Please provide a valid question."
    
    try:
        # Invoke RAG chain
        result = rag_pipeline.invoke({"input": user_message})
        bot_response = result.get("answer", "I'm sorry, I couldn't generate a response.")
        
        return str(bot_response)
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return "An error occurred while processing your request. Please try again."


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
