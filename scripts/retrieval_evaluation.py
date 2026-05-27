"""
Retrieval Performance Evaluation for Medical Chatbot
This script evaluates the performance of the RAG system using various metrics
and generates visualizations and tables for analysis.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from src.helper import initialize_embeddings
from src.prompt import get_system_prompt_template

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = "medical-chatbot"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

class RetrievalEvaluator:
    def __init__(self):
        """Initialize the evaluator with the RAG system components"""
        # Initialize embeddings
        self.embedding_model = initialize_embeddings()
        
        # Connect to Pinecone vector store
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=self.embedding_model
        )
        
        # Initialize Groq chat model
        self.llm_model = ChatGroq(
            model=GROQ_MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0
        )
        
        # Create prompt template
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", get_system_prompt_template()),
            ("human", "{input}"),
        ])
        
        # Create document chain and RAG pipeline
        document_chain = create_stuff_documents_chain(self.llm_model, chat_prompt)
        self.rag_pipeline = create_retrieval_chain(self.vector_store.as_retriever(), document_chain)
        
        # Test queries for evaluation
        self.test_queries = [
            "What are the symptoms of diabetes?",
            "How to treat hypertension?",
            "Common side effects of antibiotics",
            "What causes fever?",
            "How to prevent heart disease?",
            "Treatment for migraines",
            "Symptoms of COVID-19",
            "What is asthma?",
            "How to lower cholesterol?",
            "Treatment for depression"
        ]
        
        # Ground truth relevant documents (this would typically come from manual annotation)
        self.ground_truth = {
            "What are the symptoms of diabetes?": ["diabetes", "symptoms", "blood sugar"],
            "How to treat hypertension?": ["hypertension", "treatment", "medication", "blood pressure"],
            "Common side effects of antibiotics": ["antibiotics", "side effects", "adverse reactions"],
            "What causes fever?": ["fever", "causes", "infection", "temperature"],
            "How to prevent heart disease?": ["heart disease", "prevention", "cardiovascular"],
            "Treatment for migraines": ["migraine", "treatment", "headache", "pain"],
            "Symptoms of COVID-19": ["COVID-19", "symptoms", "coronavirus", "respiratory"],
            "What is asthma?": ["asthma", "definition", "respiratory", "condition"],
            "How to lower cholesterol?": ["cholesterol", "lower", "diet", "lipids"],
            "Treatment for depression": ["depression", "treatment", "mental health", "therapy"]
        }

    def evaluate_retrieval_performance(self, k_values=[1, 3, 5, 10]):
        """
        Evaluate retrieval performance with different k values
        
        Args:
            k_values: List of k values to test for top-k retrieval
            
        Returns:
            DataFrame with performance metrics
        """
        results = []
        
        for k in k_values:
            avg_precision = 0
            avg_recall = 0
            avg_f1 = 0
            avg_latency = 0
            queries_tested = 0
            
            for query in self.test_queries:
                # Time the retrieval
                start_time = time.time()
                
                # Get retrieved documents
                retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
                retrieved_docs = retriever.invoke(query)
                
                latency = time.time() - start_time
                
                # Check against ground truth
                ground_truth_keywords = self.ground_truth.get(query, [])
                
                # Count relevant documents retrieved
                relevant_count = 0
                for doc in retrieved_docs:
                    doc_content = doc.page_content.lower()
                    for keyword in ground_truth_keywords:
                        if keyword.lower() in doc_content:
                            relevant_count += 1
                            break  # Count each document only once
                
                # Calculate metrics
                precision = relevant_count / len(retrieved_docs) if len(retrieved_docs) > 0 else 0
                recall = relevant_count / max(len(ground_truth_keywords), 1)  # Assuming perfect recall would be len(ground_truth_keywords)
                
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Accumulate metrics
                avg_precision += precision
                avg_recall += recall
                avg_f1 += f1
                avg_latency += latency
                queries_tested += 1
            
            # Average metrics across all queries
            avg_precision /= queries_tested
            avg_recall /= queries_tested
            avg_f1 /= queries_tested
            avg_latency /= queries_tested
            
            results.append({
                'k_value': k,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1_score': avg_f1,
                'avg_latency_ms': avg_latency * 1000,  # Convert to milliseconds
                'queries_tested': queries_tested
            })
        
        return pd.DataFrame(results)

    def evaluate_similarity_scores(self, query, k=5):
        """
        Evaluate similarity scores of retrieved documents
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            DataFrame with document similarities
        """
        # Using similarity search with scores
        retriever = self.vector_store.similarity_search_with_score(query, k=k)
        
        similarities = []
        for i, (doc, score) in enumerate(retriever):
            similarities.append({
                'rank': i + 1,
                'similarity_score': score,
                'document_content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'source': doc.metadata.get('source', 'Unknown')
            })
        
        return pd.DataFrame(similarities)

    def plot_performance_metrics(self, df_results):
        """
        Plot performance metrics comparison
        
        Args:
            df_results: DataFrame with performance metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision vs K
        axes[0, 0].plot(df_results['k_value'], df_results['avg_precision'], marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_title('Average Precision vs K Value', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('K Value')
        axes[0, 0].set_ylabel('Average Precision')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall vs K
        axes[0, 1].plot(df_results['k_value'], df_results['avg_recall'], marker='s', color='orange', linewidth=2, markersize=8)
        axes[0, 1].set_title('Average Recall vs K Value', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('K Value')
        axes[0, 1].set_ylabel('Average Recall')
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score vs K
        axes[1, 0].plot(df_results['k_value'], df_results['avg_f1_score'], marker='^', color='green', linewidth=2, markersize=8)
        axes[1, 0].set_title('Average F1-Score vs K Value', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('K Value')
        axes[1, 0].set_ylabel('Average F1-Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Latency vs K
        axes[1, 1].plot(df_results['k_value'], df_results['avg_latency_ms'], marker='d', color='red', linewidth=2, markersize=8)
        axes[1, 1].set_title('Average Latency vs K Value', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('K Value')
        axes[1, 1].set_ylabel('Average Latency (ms)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_similarity_distribution(self, query, k=5):
        """
        Plot distribution of similarity scores
        
        Args:
            query: Query string
            k: Number of documents to retrieve
        """
        similarities_df = self.evaluate_similarity_scores(query, k)
        
        plt.figure(figsize=(10, 6))
        plt.bar(similarities_df['rank'], similarities_df['similarity_score'], color='skyblue', edgecolor='navy')
        plt.title(f'Distribution of Similarity Scores for Query: "{query[:50]}..."', fontsize=14, fontweight='bold')
        plt.xlabel('Rank of Retrieved Document')
        plt.ylabel('Similarity Score')
        plt.xticks(similarities_df['rank'])
        plt.grid(axis='y', alpha=0.3)
        plt.show()

def main():
    """Main function to run the evaluation"""
    print("Initializing Retrieval Evaluator...")
    evaluator = RetrievalEvaluator()
    
    print("\nEvaluating retrieval performance with different k values...")
    results_df = evaluator.evaluate_retrieval_performance(k_values=[1, 3, 5, 10])
    
    print("\nPerformance Results:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    
    # Display results as a styled table
    styled_results = results_df.style.format({
        'avg_precision': '{:.3f}',
        'avg_recall': '{:.3f}',
        'avg_f1_score': '{:.3f}',
        'avg_latency_ms': '{:.2f}'
    }).background_gradient(subset=['avg_precision', 'avg_recall', 'avg_f1_score'], cmap='Blues') \
      .background_gradient(subset=['avg_latency_ms'], cmap='Reds_r')
    
    print("\nStyled Results Table:")
    display(styled_results)  # This works in Jupyter notebooks
    
    # Plot performance metrics
    print("\nGenerating performance plots...")
    evaluator.plot_performance_metrics(results_df)
    
    # Example similarity distribution for a sample query
    sample_query = "What are the symptoms of diabetes?"
    print(f"\nGenerating similarity distribution for query: '{sample_query}'")
    evaluator.plot_similarity_distribution(sample_query, k=5)
    
    # Detailed similarity scores for sample query
    print(f"\nDetailed similarity scores for query: '{sample_query}'")
    similarities_df = evaluator.evaluate_similarity_scores(sample_query, k=5)
    print(similarities_df[['rank', 'similarity_score', 'source']].to_string(index=False))

if __name__ == "__main__":
    main()