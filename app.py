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

app = Flask(__name__)

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

if sys.version_info >= (3, 12):
    logger.warning(
        "Running on Python %s.%s — some dependencies may be incompatible. "
        "Use Python 3.11 or Docker on Render.",
        sys.version_info.major,
        sys.version_info.minor,
    )


def get_fallback_llm():
    """Lightweight Groq client (no embeddings) — always fast on Render."""
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
    """Initialize full RAG pipeline (heavy — runs in background on startup)."""
    global rag_pipeline, rag_init_error, rag_init_stage

    if rag_pipeline is not None:
        return

    with rag_init_lock:
        if rag_pipeline is not None:
            return

        try:
            rag_init_stage = "loading_imports"
            logger.info("RAG init: loading imports")

            from pinecone import Pinecone
            from langchain_pinecone import PineconeVectorStore
            from langchain_groq import ChatGroq
            from langchain.chains import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain_core.prompts import ChatPromptTemplate
            from src.helper import initialize_embeddings

            if not PINECONE_API_KEY:
                raise RuntimeError("Missing PINECONE_API_KEY")
            if not GROQ_API_KEY:
                raise RuntimeError("Missing GROQ_API_KEY")

            os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
            os.environ["GROQ_API_KEY"] = GROQ_API_KEY

            rag_init_stage = "loading_embeddings"
            logger.info("RAG init: loading embeddings")
            embedding_model = initialize_embeddings()

            rag_init_stage = "connecting_pinecone"
            logger.info("RAG init: connecting Pinecone")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)
            vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

            document_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3},
            )

            rag_init_stage = "building_chain"
            logger.info("RAG init: building chain")
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
            rag_init_stage = "ready"
            logger.info("RAG pipeline ready")

        except Exception as exc:
            rag_init_error = str(exc)
            rag_init_stage = "failed"
            logger.exception("RAG init failed: %s", rag_init_error)


def start_rag_init_background():
    """Start RAG init once in a background thread (does not block HTTP)."""
    global rag_init_thread, rag_init_stage

    if rag_pipeline is not None:
        return

    with rag_init_lock:
        if rag_pipeline is not None:
            return
        if rag_init_thread is not None and rag_init_thread.is_alive():
            return
        rag_init_stage = "starting"
        rag_init_thread = threading.Thread(target=init_rag, daemon=True)
        rag_init_thread.start()
        logger.info("RAG init started in background")


@app.route("/health")
def health():
    start_rag_init_background()
    return {
        "rag_pipeline_ready": rag_pipeline is not None,
        "rag_init_stage": rag_init_stage,
        "rag_init_thread_alive": bool(rag_init_thread and rag_init_thread.is_alive()),
        "rag_init_error": rag_init_error,
        "fallback_available": bool(GROQ_API_KEY),
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

    start_rag_init_background()

    # Full RAG path when ready
    if rag_pipeline is not None:
        try:
            result = rag_pipeline.invoke({"input": user_input})
            if isinstance(result, dict):
                answer = result.get("answer") or result.get("result") or ""
                if answer:
                    return answer
            return str(result)
        except Exception:
            logger.exception("RAG invoke failed; using fallback")

    # Fast path — always return 200 text (no gunicorn HTML 500)
    try:
        answer = answer_with_fallback(user_input)
        if rag_pipeline is None and rag_init_stage != "ready":
            prefix = "[Quick mode — document search still loading] "
            return prefix + answer
        return answer
    except Exception:
        logger.exception("Fallback failed")
        if rag_init_error:
            return f"Service error: {rag_init_error}", 500
        return "Sorry, I could not generate a response right now.", 500


# Start background RAG load as soon as the app imports (gunicorn worker).
start_rag_init_background()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
