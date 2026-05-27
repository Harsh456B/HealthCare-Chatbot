"""
Medical Chatbot Application
A Flask-based RAG chatbot for medical queries using LangChain, Pinecone, and Groq
"""

import os
import threading
import logging
from flask import Flask, render_template, request
from dotenv import load_dotenv
from src.prompt import get_system_prompt_template
    raise RuntimeError(
        f"Detected Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        "This application requires Python 3.11. Please set your host to use Python 3.11.\n"
        "If you are deploying to Render, either: (1) set the service runtime to Python 3.11,\n"
        "or (2) deploy using the provided Dockerfile which pins Python 3.11."
    )

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

load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = "medical-chatbot"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# Globals for lazy initialization
rag_pipeline = None
rag_init_lock = threading.Lock()
rag_init_error = None

# Basic logger
logger = logging.getLogger("medical_chatbot")
logging.basicConfig(level=logging.INFO)

# Warn (but do not fail) on unsupported Python versions so we can still start
import sys
if sys.version_info >= (3, 12):
    logger.warning(
        "Running on Python %s.%s — some dependencies may be incompatible. "
        "If you see pydantic/typing errors, deploy with Python 3.11 or use the Dockerfile.",
        sys.version_info.major,
        sys.version_info.minor,
    )


def init_rag():
    """Lazy initialize LangChain, Pinecone, and Groq components.

    This avoids importing heavy/typing-sensitive libraries at module import
    time which can crash under newer Python runtimes on some hosts.
    """
    global rag_pipeline, rag_init_error

    if rag_pipeline is not None or rag_init_error is not None:
        return

    with rag_init_lock:
        if rag_pipeline is not None or rag_init_error is not None:
            return

        try:
            # Import heavy deps here
            import pinecone
            from langchain.vectorstores import Pinecone as PineconeVectorStore
            from langchain_groq import ChatGroq
            from langchain.chains import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain_core.prompts import ChatPromptTemplate
            from src.helper import initialize_embeddings

            # Set environment variables for downstream libs
            if PINECONE_API_KEY:
                os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
            if GROQ_API_KEY:
                os.environ["GROQ_API_KEY"] = GROQ_API_KEY

            # Initialize embeddings
            embedding_model = initialize_embeddings()

            # Connect to Pinecone (if API key exists)
            if PINECONE_API_KEY:
                pinecone.init(api_key=PINECONE_API_KEY)

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

            logger.info("RAG pipeline initialized successfully")

        except Exception as e:
            rag_init_error = str(e)
            logger.exception("Failed to initialize RAG pipeline: %s", rag_init_error)
