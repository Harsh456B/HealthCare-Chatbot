#!/usr/bin/env python
"""
Script to run the retrieval evaluation and generate summary results
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.retrieval_evaluation import RetrievalEvaluator

def main():
    print("Starting Retrieval Performance Evaluation...")
    print("="*50)
    
    # Initialize evaluator
    try:
        evaluator = RetrievalEvaluator()
        print("✓ Evaluator initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize evaluator: {str(e)}")
        print("Please ensure Pinecone index exists and API keys are correctly configured")
        return
    
    # Run evaluation
    print("\nRunning evaluation with different k values...")
    results_df = evaluator.evaluate_retrieval_performance(k_values=[1, 3, 5, 10])
    
    print("\n" + "="*50)
    print("RETRIEVAL PERFORMANCE RESULTS")
    print("="*50)
    
    # Print formatted results
    for _, row in results_df.iterrows():
        print(f"K = {int(row['k_value']):2d}: "
              f"Prec: {row['avg_precision']:.3f}, "
              f"Rec: {row['avg_recall']:.3f}, "
              f"F1: {row['avg_f1_score']:.3f}, "
              f"Lat: {row['avg_latency_ms']:.2f}ms")
    
    print("\n" + "="*50)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*50)
    
    # Find best k value based on F1 score
    best_k_idx = results_df['avg_f1_score'].idxmax()
    best_k = results_df.loc[best_k_idx, 'k_value']
    best_precision = results_df.loc[best_k_idx, 'avg_precision']
    best_recall = results_df.loc[best_k_idx, 'avg_recall']
    best_f1 = results_df.loc[best_k_idx, 'avg_f1_score']
    best_latency = results_df.loc[best_k_idx, 'avg_latency_ms']
    
    print(f"Optimal K Value: {int(best_k)}")
    print(f"Achieved Metrics:")
    print(f"  - Precision:  {best_precision:.3f}")
    print(f"  - Recall:     {best_recall:.3f}")
    print(f"  - F1-Score:   {best_f1:.3f}")
    print(f"  - Latency:    {best_latency:.2f} ms")
    
    print(f"\nRecommendation:")
    print(f"The optimal setting is k={int(best_k)} which provides the best balance")
    print(f"between retrieval accuracy and response time.")
    
    if best_k <= 3:
        print("With this setting, the system maintains low latency while achieving good retrieval quality.")
    elif best_k <= 5:
        print("With this setting, the system achieves good balance between recall and performance.")
    else:
        print("With this setting, the system prioritizes recall over latency, suitable for critical medical queries.")
    
    # Save results to CSV
    results_df.to_csv('retrieval_evaluation_results.csv', index=False)
    print(f"\nResults saved to 'retrieval_evaluation_results.csv'")
    
    # Generate visualization
    print("Generating performance plots...")
    evaluator.plot_performance_metrics(results_df)
    
    # Show similarity distribution for sample query
    sample_query = "What are the symptoms of diabetes?"
    print(f"Generating similarity distribution for: '{sample_query}'")
    evaluator.plot_similarity_distribution(sample_query, k=5)
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()