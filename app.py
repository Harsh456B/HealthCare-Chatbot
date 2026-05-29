"""
Medical Chatbot Application
A Streamlit-based RAG chatbot for medical queries using LangChain, Pinecone, and Groq
"""

import os
import sys
import threading
import logging
import streamlit as st
from dotenv import load_dotenv
from src.prompt import get_system_prompt_template

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = "medical-chatbot"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

rag_pipeline = None
rag_init_lock = threading.Lock()
rag_init_error = None
rag_init_stage = "not_started"
rag_init_thread = None

fallback_llm = None
fallback_lock = threading.Lock()

logger = logging.getLogger("medical_chatbot")
logging.basicConfig(level=logging.INFO)


def get_fallback_llm():
    """Lightweight Groq client (no embeddings) — always fast."""
    global fallback_llm
    if fallback_llm is not None:
        return fallback_llm
    with fallback_lock:
        if fallback_llm is not None:
            return fallback_llm
        if not GROQ_API_KEY:
            raise RuntimeError("Missing GROQ_API_KEY")
        from langchain_groq import ChatGroq

        fallback_llm = ChatGroq(
            model=GROQ_MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0,
        )
        return fallback_llm


def answer_with_fallback(user_input: str) -> str:
    """Answer without Pinecone/RAG when pipeline is not ready."""
    llm = get_fallback_llm()
    system = (
        "You are a helpful medical assistant. Answer clearly in at most 3 sentences. "
        "If you are unsure, say you are not certain and suggest consulting a doctor."
    )
    response = llm.invoke([
        ("system", system),
        ("human", user_input),
    ])
    return getattr(response, "content", str(response))


def init_rag():
    """RAG pipeline disabled for Python 3.14 compatibility - using fallback only."""
    global rag_pipeline, rag_init_error, rag_init_stage
    rag_pipeline = None
    rag_init_error = "RAG disabled for Python 3.14 compatibility"
    rag_init_stage = "disabled"
    logger.info("RAG pipeline disabled - using fallback LLM only")


def start_rag_init_background():
    """RAG init disabled for Python 3.14 compatibility."""
    global rag_pipeline, rag_init_stage
    rag_pipeline = None
    rag_init_stage = "disabled"
    logger.info("RAG init disabled - using fallback LLM only")


# Streamlit UI - Exact replica of original Flask UI
st.set_page_config(
    page_title="Medical Chatbot", 
    page_icon="🏥",
    layout="centered"
)

# Custom CSS to exactly match original Flask UI
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f5;
    }
    .main {
        padding-top: 0;
        padding-bottom: 0;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    .chat-header {
        background: #4a90e2;
        color: white;
        padding: 20px;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .chat-header img {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        border: 2px solid white;
    }
    .chat-header h2 {
        margin: 0;
        font-size: 18px;
    }
    .chat-header p {
        margin: 5px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
    }
    .chat-messages {
        padding: 20px;
        background: #f9f9f9;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
    }
    .stChatMessage {
        background-color: transparent;
        border-radius: 0;
        padding: 0;
        margin-bottom: 15px;
    }
    .stChatMessage[data-testid="st-chat-message"] {
        background-color: transparent;
    }
    .stChatMessage[data-testid="st-chat-message"] > div {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Header - Exact replica of original
st.markdown("""
<div class="chat-header">
    <img src="https://cdn-icons-png.flaticon.com/512/387/387569.png" alt="Medical Bot">
    <div>
        <h2>Medical Chatbot</h2>
        <p>Ask me anything about medical topics!</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# RAG disabled for Python 3.14 compatibility
start_rag_init_background()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Using fallback LLM only (RAG disabled for Python 3.14 compatibility)
                response = answer_with_fallback(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.markdown('</div>', unsafe_allow_html=True)

# Hide default Streamlit elements
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)
