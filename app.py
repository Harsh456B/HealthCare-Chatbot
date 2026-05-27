"""
Medical Chatbot Application
A Flask-based RAG chatbot for medical queries using LangChain, Pinecone, and Groq
"""

import os
import sys
import threading
import logging
from flask import Flask, render_template, request
from dotenv import load_dotenv
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
rag_init_started = False
rag_init_thread = None

# Basic logger
logger = logging.getLogger("medical_chatbot")
logging.basicConfig(level=logging.INFO)

if sys.version_info >= (3, 12):
    logger.warning(
        "Running on Python %s.%s — some dependencies may be incompatible. "
        "If you see pydantic/typing errors, deploy with Python 3.11 or use the Dockerfile.",
        sys.version_info.major,
        sys.version_info.minor,
    )


def init_rag():
    """Lazy initialize LangChain, Pinecone, and Groq components."""
    global rag_pipeline, rag_init_error

    # If we already have a pipeline, nothing to do.
    # If we had a previous init error, we allow retries (e.g. after fixing env vars).
    if rag_pipeline is not None:
        return

    with rag_init_lock:
        if rag_pipeline is not None:
            return

        try:
            # Prefer the modern Pinecone + LangChain integration (works with pinecone>=6).
            # Fall back to the legacy LangChain vectorstore if needed.
            PineconeVectorStore = None
            pc = None
            index = None

            try:
                from pinecone import Pinecone  # pinecone>=3
                from langchain_pinecone import PineconeVectorStore as LCPineconeVectorStore

                pc = Pinecone(api_key=PINECONE_API_KEY)
                index = pc.Index(PINECONE_INDEX_NAME)
                PineconeVectorStore = LCPineconeVectorStore
            except Exception:
                import pinecone  # legacy client
                try:
                    # LangChain >= 0.1 (community split)
                    from langchain_community.vectorstores import Pinecone as PineconeVectorStore
                except Exception:
                    # Older LangChain
                    from langchain.vectorstores import Pinecone as PineconeVectorStore
            from langchain_groq import ChatGroq
            from langchain.chains import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain_core.prompts import ChatPromptTemplate
            from src.helper import initialize_embeddings

            if not PINECONE_API_KEY:
                raise RuntimeError("Missing PINECONE_API_KEY")
            if not GROQ_API_KEY:
                raise RuntimeError("Missing GROQ_API_KEY")

            if PINECONE_API_KEY:
                os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
            if GROQ_API_KEY:
                os.environ["GROQ_API_KEY"] = GROQ_API_KEY

            embedding_model = initialize_embeddings()

            if pc is not None and index is not None and PineconeVectorStore is not None:
                # langchain-pinecone path
                vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
            else:
                # Legacy langchain_community path
                try:
                    pinecone.init(api_key=PINECONE_API_KEY)
                except Exception:
                    pass

                vector_store = PineconeVectorStore.from_existing_index(
                    index_name=PINECONE_INDEX_NAME,
                    embedding=embedding_model,
                )

            document_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3},
            )

            llm_model = ChatGroq(
                model=GROQ_MODEL_NAME,
                api_key=GROQ_API_KEY,
                temperature=0,
            )

            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", get_system_prompt_template()),
                ("human", "{input}"),
            ])

            document_chain = create_stuff_documents_chain(llm_model, chat_prompt)
            rag_pipeline = create_retrieval_chain(document_retriever, document_chain)
            rag_init_error = None

            logger.info("RAG pipeline initialized successfully")

        except Exception as exc:
            rag_init_error = str(exc)
            logger.exception("Failed to initialize RAG pipeline: %s", rag_init_error)


def ensure_rag_init_started():
    """Kick off RAG initialization in background once."""
    global rag_init_started, rag_init_thread
    if rag_pipeline is not None:
        return

    # If a previous init thread died without producing a pipeline/error,
    # allow a clean retry.
    if (
        rag_init_started
        and rag_pipeline is None
        and rag_init_error is None
        and rag_init_thread is not None
        and not rag_init_thread.is_alive()
    ):
        rag_init_started = False
        rag_init_thread = None

    # Do not block requests if init is already running (init_rag holds this lock
    # during heavy model/vector initialization).
    acquired = rag_init_lock.acquire(blocking=False)
    if not acquired:
        return
    try:
        if rag_pipeline is not None or rag_init_started:
            return
        rag_init_started = True
        rag_init_thread = threading.Thread(target=init_rag, daemon=True)
        rag_init_thread.start()
    finally:
        rag_init_lock.release()


@app.route("/health")
def health():
    """Lightweight health/debug endpoint."""
    ensure_rag_init_started()
    return {
        "rag_pipeline_ready": rag_pipeline is not None,
        "rag_init_started": rag_init_started,
        "rag_init_thread_alive": bool(rag_init_thread and rag_init_thread.is_alive()),
        "rag_init_error": rag_init_error,
        "pinecone_index": PINECONE_INDEX_NAME,
        "groq_model": GROQ_MODEL_NAME,
    }


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def get_response():
    user_input = request.form.get("msg", "").strip()
    if not user_input:
        return "Please enter a message.", 400

    ensure_rag_init_started()

    if rag_pipeline is None and rag_init_error is None:
        return (
            "Model is warming up. Please retry in 20-60 seconds.",
            503,
        )

    if rag_init_error:
        logger.error("RAG init error: %s", rag_init_error)
        return (
            "Unable to process requests at this time. "
            "Check that PINECONE_API_KEY, GROQ_API_KEY, and the index are configured.",
            500,
        )

    try:
        result = rag_pipeline.invoke({"input": user_input})
        if isinstance(result, dict):
            answer = result.get("answer") or result.get("result") or ""
            if answer:
                return answer
            return "Sorry, I couldn't generate a response right now.", 500
        return str(result)
    except Exception:
        logger.exception("Failed to generate bot response")
        return "Sorry, I couldn't generate a response right now.", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
