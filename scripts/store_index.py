"""
Script to create and populate Pinecone vector index with medical documents
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

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_DIRECTORY = "data/"
INDEX_NAME = "medical-chatbot"
EMBEDDING_DIMENSION = 384
CLOUD_PROVIDER = "aws"
REGION = "us-east-1"

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


def create_pinecone_index(pc_client: Pinecone, index_name: str) -> None:
    """
    Create a new Pinecone index if it doesn't exist
    
    Args:
        pc_client: Pinecone client instance
        index_name: Name of the index to create
    """
    if not pc_client.has_index(index_name):
        print(f"Creating new index: {index_name}")
        pc_client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD_PROVIDER, region=REGION),
        )
        print(f"Index '{index_name}' created successfully!")
    else:
        print(f"Index '{index_name}' already exists.")


def main():
    """Main function to process documents and store in Pinecone"""
    print("Starting document processing...")
    
    # Step 1: Load PDF documents
    print(f"Loading PDF files from {DATA_DIRECTORY}...")
    raw_documents = load_pdf_documents(DATA_DIRECTORY)
    print(f"Loaded {len(raw_documents)} documents")
    
    # Step 2: Clean document metadata
    print("Cleaning document metadata...")
    cleaned_documents = clean_document_metadata(raw_documents)
    print(f"Cleaned {len(cleaned_documents)} documents")
    
    # Step 3: Split documents into chunks
    print("Splitting documents into chunks...")
    document_chunks = split_documents_into_chunks(cleaned_documents)
    print(f"Created {len(document_chunks)} text chunks")
    
    # Step 4: Initialize embeddings
    print("Initializing embeddings model...")
    embeddings_model = initialize_embeddings()
    print("Embeddings model ready")
    
    # Step 5: Initialize Pinecone client
    print("Connecting to Pinecone...")
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    
    # Step 6: Create index if needed
    create_pinecone_index(pinecone_client, INDEX_NAME)
    
    # Step 7: Store documents in Pinecone
    print(f"Storing documents in Pinecone index '{INDEX_NAME}'...")
    vector_store = PineconeVectorStore.from_documents(
        documents=document_chunks,
        index_name=INDEX_NAME,
        embedding=embeddings_model
    )
    
    print("Documents successfully stored in Pinecone!")
    print("You can now run the Flask application.")


if __name__ == "__main__":
    main()
