import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the evaluation results
df = pd.read_csv('retrieval_evaluation_results.csv')

# Create a single comprehensive graph with all metrics
fig, ax1 = plt.subplots(figsize=(14, 8))

# Set width of bars
bar_width = 0.2
bars = np.arange(len(df['k_value']))

# Create bars for precision, recall, and F1-score on primary y-axis
bars1 = ax1.bar(bars - bar_width, df['avg_precision'], bar_width, label='Precision', color='skyblue', edgecolor='navy', alpha=0.8)
bars2 = ax1.bar(bars, df['avg_recall'], bar_width, label='Recall', color='lightcoral', edgecolor='darkred', alpha=0.8)
bars3 = ax1.bar(bars + bar_width, df['avg_f1_score'], bar_width, label='F1-Score', color='lightgreen', edgecolor='darkgreen', alpha=0.8)

# Set labels and title for primary axis
ax1.set_xlabel('K Value', fontsize=12)
ax1.set_ylabel('Precision / Recall / F1-Score', fontsize=12, color='black')
ax1.set_title('Comprehensive Retrieval Performance Evaluation - All Metrics', fontsize=16, fontweight='bold', pad=20)

# Set x-axis ticks and labels
ax1.set_xticks(bars)
ax1.set_xticklabels(df['k_value'])
ax1.grid(axis='y', alpha=0.3)

# Create secondary y-axis for latency
ax2 = ax1.twinx()
bars4 = ax2.plot(bars, df['avg_latency_ms'], 'ro-', linewidth=2.5, markersize=8, label='Latency (ms)', color='purple')

# Set labels for secondary axis
ax2.set_ylabel('Latency (ms)', fontsize=12, color='purple')

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Add value labels on bars for precision, recall, and F1
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)

for bar in bars3:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# Add value labels on line for latency
for i, v in enumerate(df['avg_latency_ms']):
    ax2.annotate(f'{v:.1f}', xy=(bars[i], v), xytext=(bars[i], v + 20),
                 ha='center', va='bottom', fontsize=9, color='purple',
                 arrowprops=dict(arrowstyle='->', color='purple', lw=0.8))

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('comprehensive_retrieval_evaluation.png', dpi=300, bbox_inches='tight')
print("Single comprehensive graph saved as 'comprehensive_retrieval_evaluation.png'")

# Also create a second version with a different layout that puts latency as bars too
fig2, ax = plt.subplots(figsize=(14, 8))

# Set width of bars for all metrics
bar_width = 0.15
bars = np.arange(len(df['k_value']))

# Create bars for all metrics
bars1 = ax.bar(bars - bar_width*1.5, df['avg_precision'], bar_width, label='Precision', color='skyblue', edgecolor='navy', alpha=0.8)
bars2 = ax.bar(bars - bar_width/2, df['avg_recall'], bar_width, label='Recall', color='lightcoral', edgecolor='darkred', alpha=0.8)
bars3 = ax.bar(bars + bar_width/2, df['avg_f1_score'], bar_width, label='F1-Score', color='lightgreen', edgecolor='darkgreen', alpha=0.8)
bars4 = ax.bar(bars + bar_width*1.5, df['avg_latency_ms']/10, bar_width, label='Latency/10 (ms scaled)', color='gold', edgecolor='orange', alpha=0.8)

# Set labels and title
ax.set_xlabel('K Value', fontsize=12)
ax.set_ylabel('Metrics Values', fontsize=12)
ax.set_title('Comprehensive Retrieval Performance Evaluation - All Metrics (Bar Chart)', fontsize=16, fontweight='bold', pad=20)

# Set x-axis ticks and labels
ax.set_xticks(bars)
ax.set_xticklabels(df['k_value'])
ax.grid(axis='y', alpha=0.3)
ax.legend()

# Add value labels on bars
for bars_set, col_name in [(bars1, 'avg_precision'), (bars2, 'avg_recall'), (bars3, 'avg_f1_score'), (bars4, 'avg_latency_ms')]:
    for i, bar in enumerate(bars_set):
        height = bar.get_height()
        value = df[col_name].iloc[i] if 'latency' not in col_name else df['avg_latency_ms'].iloc[i]
        ax.text(bar.get_x() + bar.get_width()/2., height + (max(df[col_name])/50 if 'latency' not in col_name else max(df['avg_latency_ms']/10)/50),
                f'{value:.1f}' if 'latency' in col_name else f'{value:.2f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('comprehensive_retrieval_evaluation_bars.png', dpi=300, bbox_inches='tight')
print("Alternative bar chart saved as 'comprehensive_retrieval_evaluation_bars.png'")

plt.show()
print("Graphs generated successfully!")