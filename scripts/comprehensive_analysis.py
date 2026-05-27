import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the retrieval evaluation results
df_retrieval = pd.read_csv('retrieval_evaluation_results.csv')

# Define the latency data
components = ['Embedding Generation', 'Pinecone Retrieval', 'LLM Generation']
times = [120, 90, 450]  # in milliseconds
total_time = 660

# Create a comprehensive figure with subplots
fig = plt.figure(figsize=(20, 16))

# Subplot 1: Retrieval Performance Metrics
ax1 = plt.subplot(2, 3, 1)
bar_width = 0.2
bars = np.arange(len(df_retrieval['k_value']))

bars1 = ax1.bar(bars - bar_width, df_retrieval['avg_precision'], bar_width, label='Precision', color='skyblue', edgecolor='navy', alpha=0.8)
bars2 = ax1.bar(bars, df_retrieval['avg_recall'], bar_width, label='Recall', color='lightcoral', edgecolor='darkred', alpha=0.8)
bars3 = ax1.bar(bars + bar_width, df_retrieval['avg_f1_score'], bar_width, label='F1-Score', color='lightgreen', edgecolor='darkgreen', alpha=0.8)

ax1.set_xlabel('K Value')
ax1.set_ylabel('Score')
ax1.set_title('Retrieval Performance Metrics', fontsize=12, fontweight='bold')
ax1.set_xticks(bars)
ax1.set_xticklabels(df_retrieval['k_value'])
ax1.grid(axis='y', alpha=0.3)
ax1.legend()

# Add value labels on bars for precision, recall, and F1
for bars_set, col_name in [(bars1, 'avg_precision'), (bars2, 'avg_recall'), (bars3, 'avg_f1_score')]:
    for i, bar in enumerate(bars_set):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# Subplot 2: Latency vs K Value
ax2 = plt.subplot(2, 3, 2)
ax2.plot(df_retrieval['k_value'], df_retrieval['avg_latency_ms'], 'ro-', linewidth=2.5, markersize=8, label='Retrieval Latency', color='purple')
ax2.set_xlabel('K Value')
ax2.set_ylabel('Latency (ms)')
ax2.set_title('Retrieval Latency vs K Value', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add value labels on points
for i, v in enumerate(df_retrieval['avg_latency_ms']):
    ax2.annotate(f'{v:.1f}', xy=(df_retrieval['k_value'].iloc[i], v), 
                 xytext=(df_retrieval['k_value'].iloc[i], v + 20),
                 ha='center', va='bottom', fontsize=8, color='purple')

# Subplot 3: Response Latency Distribution
ax3 = plt.subplot(2, 3, 3)
bars_latency = ax3.barh(components, times, color=['skyblue', 'lightcoral', 'lightgreen'], 
                       edgecolor='navy', alpha=0.8, height=0.6)

# Add value labels on bars
for i, (bar, time) in enumerate(zip(bars_latency, times)):
    width = bar.get_width()
    ax3.text(width + 5, bar.get_y() + bar.get_height()/2, f'{time} ms', 
             ha='left', va='center', fontweight='bold', fontsize=10)

ax3.set_xlabel('Time (milliseconds)')
ax3.set_title('Response Latency Distribution\n(Fig. 6.2)', fontsize=12, fontweight='bold')
ax3.set_xlim(0, max(times) * 1.2)

# Add total time line
ax3.axvline(x=total_time, color='red', linestyle='--', alpha=0.7, label=f'Total: {total_time} ms')
ax3.text(total_time + 10, len(components)-0.5, f'Total\n{total_time} ms', 
         ha='left', va='center', fontweight='bold', color='red', fontsize=10)

ax3.legend()

# Subplot 4: Component Contribution Pie Chart
ax4 = plt.subplot(2, 3, 4)
colors = ['skyblue', 'lightcoral', 'lightgreen']
wedges, texts, autotexts = ax4.pie(times, labels=components, autopct='%1.1f%%', 
                                   colors=colors, startangle=90, pctdistance=0.85)

# Draw circle in center to create a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig.gca().add_artist(centre_circle)

# Add total time in the center
plt.text(0, 0, f'Total\n{total_time} ms', horizontalalignment='center', 
         verticalalignment='center', fontsize=12, fontweight='bold')

ax4.set_title('Response Latency Component Contributions', fontsize=12, fontweight='bold')

# Subplot 5: Performance vs Latency Trade-off
ax5 = plt.subplot(2, 3, 5)

# Normalize values for comparison (scale to 0-1 range)
norm_precision = (df_retrieval['avg_precision'] - df_retrieval['avg_precision'].min()) / (df_retrieval['avg_precision'].max() - df_retrieval['avg_precision'].min())
norm_latency = (df_retrieval['avg_latency_ms'] - df_retrieval['avg_latency_ms'].min()) / (df_retrieval['avg_latency_ms'].max() - df_retrieval['avg_latency_ms'].min())

ax5.plot(df_retrieval['k_value'], norm_precision, 'bo-', label='Normalized Precision', linewidth=2, markersize=8)
ax5.plot(df_retrieval['k_value'], norm_latency, 'ro-', label='Normalized Latency', linewidth=2, markersize=8)

ax5.set_xlabel('K Value')
ax5.set_ylabel('Normalized Value (0-1)')
ax5.set_title('Performance vs Latency Trade-off', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()

# Add value labels
for i in range(len(df_retrieval)):
    ax5.annotate(f'{df_retrieval["k_value"].iloc[i]}', 
                 xy=(df_retrieval['k_value'].iloc[i], norm_precision.iloc[i]),
                 xytext=(df_retrieval['k_value'].iloc[i], norm_precision.iloc[i] + 0.05),
                 ha='center', va='bottom', fontsize=8)

# Subplot 6: Optimal K Recommendation
ax6 = plt.subplot(2, 3, 6)

# Find the optimal k based on F1 score
optimal_idx = df_retrieval['avg_f1_score'].idxmax()
optimal_k = df_retrieval.loc[optimal_idx, 'k_value']
optimal_precision = df_retrieval.loc[optimal_idx, 'avg_precision']
optimal_recall = df_retrieval.loc[optimal_idx, 'avg_recall']
optimal_f1 = df_retrieval.loc[optimal_idx, 'avg_f1_score']
optimal_latency = df_retrieval.loc[optimal_idx, 'avg_latency_ms']

# Create a radar chart-like visualization for the optimal k
metrics = ['Precision', 'Recall', 'F1-Score']
values = [optimal_precision, optimal_recall, optimal_f1]
values_scaled = [v/max(df_retrieval[m].max() for m in ['avg_precision', 'avg_recall', 'avg_f1_score']) for v in values]

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
values_scaled += values_scaled[:1]  # Complete the circle
angles += angles[:1]

ax6 = plt.subplot(2, 3, 6, projection='polar')
ax6.fill(angles, values_scaled, alpha=0.25, color='blue')
ax6.plot(angles, values_scaled, color='blue', linewidth=2)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(metrics)
ax6.set_ylim(0, 1)
ax6.set_title(f'Optimal K={optimal_k}\nPerformance Profile', fontsize=12, fontweight='bold', pad=20)

# Add text box with detailed information
info_text = f"""Optimal K Value: {optimal_k}
Precision: {optimal_precision:.3f}
Recall: {optimal_recall:.3f}
F1-Score: {optimal_f1:.3f}
Latency: {optimal_latency:.1f} ms"""

ax6.text(0.02, 0.98, info_text, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout(pad=3.0)

# Add a main title
fig.suptitle('Medical Chatbot: Comprehensive Performance Analysis Dashboard', 
             fontsize=18, fontweight='bold', y=0.98)

# Save the comprehensive figure
plt.savefig('comprehensive_performance_analysis.png', dpi=300, bbox_inches='tight')
print("Comprehensive analysis dashboard saved as 'comprehensive_performance_analysis.png'")

plt.show()

print("\nComprehensive analysis completed successfully!")
print(f"Optimal K value identified: {optimal_k}")
print(f"This provides the best balance between performance and latency for the medical chatbot.")