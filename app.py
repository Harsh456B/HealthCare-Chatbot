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


# Streamlit UI
st.set_page_config(page_title="Medical Chatbot", page_icon="🏥")

st.title("🏥 Medical Chatbot")
st.write("Ask me anything about medical topics!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Start RAG initialization
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
                if rag_pipeline is not None:
                    result = rag_pipeline.invoke({"input": prompt})
                    if isinstance(result, dict):
                        response = result.get("answer") or result.get("result") or ""
                        if response:
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            response = str(result)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        response = str(result)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    # Fallback
                    response = answer_with_fallback(prompt)
                    if rag_init_stage != "ready":
                        response = "[Quick mode — document search still loading] " + response
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
