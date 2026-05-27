"""
Helper functions for document processing and embeddings
"""

import logging
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List


def load_pdf_documents(directory_path: str) -> List[Document]:
    """
    Load all PDF files from the specified directory
    
    Args:
        directory_path: Path to directory containing PDF files
        
    Returns:
        List of Document objects containing PDF content
    """
    loader = DirectoryLoader(
        directory_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    return documents


def clean_document_metadata(documents: List[Document]) -> List[Document]:
    """
    Clean document metadata to keep only essential information
    
    Args:
        documents: List of Document objects with metadata
        
    Returns:
        List of Document objects with cleaned metadata
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
    Split documents into smaller chunks for better processing
    
    Args:
        documents: List of Document objects to split
        
    Returns:
        List of Document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


def initialize_embeddings():
    """
    Initialize embeddings (FastEmbed on CPU when available, else HuggingFace).
    Must match index model: sentence-transformers/all-MiniLM-L6-v2 (384-dim).
    """
    log = logging.getLogger("medical_chatbot")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    try:
        from langchain_community.embeddings import FastEmbedEmbeddings

        log.info("Loading FastEmbed model: %s", model_name)
        embeddings = FastEmbedEmbeddings(
            model_name=model_name,
            cache_dir=os.getenv("HF_HOME", "/app/.cache/huggingface"),
        )
        log.info("FastEmbed model loaded.")
        return embeddings
    except Exception as fastembed_err:
        log.warning("FastEmbed unavailable (%s), using HuggingFaceEmbeddings", fastembed_err)

    log.info("Loading HuggingFace embeddings on CPU: %s", model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    log.info("HuggingFace embeddings loaded.")
    return embeddings
