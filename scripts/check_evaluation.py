import sys
import os
sys.path.append('.')

from research.retrieval_evaluation import RetrievalEvaluator
import pandas as pd

print('Starting Retrieval Performance Evaluation...')
evaluator = RetrievalEvaluator()
print('✓ Evaluator initialized successfully')

print('\nRunning evaluation with different k values...')
results_df = evaluator.evaluate_retrieval_performance(k_values=[1, 3, 5, 10])

print('\n' + '='*50)
print('RETRIEVAL PERFORMANCE RESULTS')
print('='*50)

for _, row in results_df.iterrows():
    print(f"K = {int(row['k_value']):2d}: "
          f"Pres: {row['avg_precision']:.3f}, "
          f"Rec: {row['avg_recall']:.3f}, "
          f"F1: {row['avg_f1_score']:.3f}, "
          f"Lat: {row['avg_latency_ms']:.2f}ms")

# Save results to CSV
results_df.to_csv('retrieval_evaluation_results.csv', index=False)
print(f'\nResults saved to retrieval_evaluation_results.csv')

print('\nEvaluation completed successfully!')