"""
Helper functions for document processing and embeddings
"""

import logging
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
    Initialize HuggingFace embeddings model
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    logger = logging.getLogger("medical_chatbot")
    logger.info("Loading embedding model on CPU (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded.")
    return embeddings
