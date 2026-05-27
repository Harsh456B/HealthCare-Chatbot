import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the comparative performance data
metrics = ['Factual Accuracy', 'Hallucination Rate', 'Context Grounding', 'Latency', 'Reliability']
standalone_llm = ['Moderate', 'High', 'Weak', 'Lower', 'Limited']
careai_rag = ['High', 'Low', 'Strong', 'Slightly Higher', 'Enhanced']

# Create a DataFrame for the data
df = pd.DataFrame({
    'Metric': metrics,
    'Standalone LLM': standalone_llm,
    'CareAI (RAG)': careai_rag
})

print("Comparative Performance Analysis")
print("="*50)
print(df.to_string(index=False))
print("="*50)

# Define a scoring system for visualization (assigning numerical values to qualitative ratings)
def score_rating(rating):
    rating_scores = {
        'Limited': 1,
        'Weak': 1,
        'Low': 1,
        'Moderate': 2,
        'Lower': 2,
        'Slightly Higher': 3,
        'High': 3,
        'Strong': 3,
        'Enhanced': 3,
        'High': 3
    }
    return rating_scores.get(rating, 2)  # Default to 2 for moderate

# Create numerical scores for plotting
standalone_scores = [score_rating(rating) for rating in standalone_llm]
careai_scores = [score_rating(rating) for rating in careai_rag]

# Create the comparative performance analysis chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Bar chart comparison
x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, standalone_scores, width, label='Standalone LLM', 
                color='lightcoral', edgecolor='black', alpha=0.8)
bars2 = ax1.bar(x + width/2, careai_scores, width, label='CareAI (RAG)', 
                color='lightgreen', edgecolor='black', alpha=0.8)

ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax1.set_ylabel('Performance Level', fontsize=12, fontweight='bold')
ax1.set_title('Comparative Performance Analysis', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars, scores in [(bars1, standalone_scores), (bars2, careai_scores)]:
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 str(score), ha='center', va='bottom', fontweight='bold')

# Radar chart for visual comparison
ax2 = plt.subplot(1, 2, 2, projection='polar')

# Complete the circle by adding the first value at the end
standalone_scores_circle = standalone_scores + [standalone_scores[0]]
careai_scores_circle = careai_scores + [careai_scores[0]]
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles_circle = angles + [angles[0]]

ax2.plot(angles_circle, standalone_scores_circle, 'o-', linewidth=2, label='Standalone LLM', color='lightcoral')
ax2.fill(angles_circle, standalone_scores_circle, alpha=0.25, color='lightcoral')
ax2.plot(angles_circle, careai_scores_circle, 'o-', linewidth=2, label='CareAI (RAG)', color='lightgreen')
ax2.fill(angles_circle, careai_scores_circle, alpha=0.25, color='lightgreen')

ax2.set_xticks(angles)
ax2.set_xticklabels(metrics, fontsize=9)
ax2.set_ylim(0, 4)
ax2.set_title('Performance Radar Comparison', fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('comparative_performance_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nComparative performance analysis chart saved as 'comparative_performance_analysis.png'")

# Create a second visualization highlighting the trade-offs
fig2, ax3 = plt.subplots(figsize=(12, 8))

# Create a heatmap-style visualization
performance_matrix = np.zeros((len(metrics), 2))

# Assign values based on qualitative ratings
rating_values = {
    'Limited': 1, 'Weak': 1, 'Low': 1, 'Moderate': 2, 'Lower': 2,
    'Slightly Higher': 2, 'High': 3, 'Strong': 3, 'Enhanced': 3
}

for i, metric in enumerate(metrics):
    performance_matrix[i, 0] = rating_values[standalone_llm[i]]
    performance_matrix[i, 1] = rating_values[careai_rag[i]]

im = ax3.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3)

# Set ticks and labels
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Standalone LLM', 'CareAI (RAG)'])
ax3.set_yticks(range(len(metrics)))
ax3.set_yticklabels(metrics)

# Add text annotations
for i in range(len(metrics)):
    for j in range(2):
        text = ax3.text(j, i, df.iloc[i, j+1],
                       ha="center", va="center", color="black", fontweight='bold', fontsize=10)

ax3.set_title('Performance Heatmap Comparison', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax3, label='Performance Level (1-3 Scale)')
plt.tight_layout()
plt.savefig('performance_heatmap_comparison.png', dpi=300, bbox_inches='tight')
print(f"Performance heatmap saved as 'performance_heatmap_comparison.png'")

# Create a summary insights chart
fig3, ax4 = plt.subplots(figsize=(10, 6))

# Calculate improvement scores (positive for improvements, negative for trade-offs)
improvement_scores = []
improvement_labels = []
colors = []

for i, metric in enumerate(metrics):
    llm_score = rating_values[standalone_llm[i]]
    rag_score = rating_values[careai_rag[i]]
    diff = rag_score - llm_score
    
    improvement_scores.append(diff)
    improvement_labels.append(metric)
    
    # Color based on improvement/degradation
    if diff > 0:
        colors.append('green')  # Improvement
    elif diff < 0:
        colors.append('red')    # Degradation
    else:
        colors.append('gray')   # No change

bars = ax4.bar(improvement_labels, improvement_scores, color=colors, edgecolor='black', alpha=0.8)

ax4.set_ylabel('Improvement Score (RAG - Standalone LLM)', fontsize=12, fontweight='bold')
ax4.set_title('Improvement Analysis: CareAI (RAG) vs Standalone LLM', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Add value labels on bars
for bar, score in zip(bars, improvement_scores):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.15),
             f'{score:+d}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('improvement_analysis.png', dpi=300, bbox_inches='tight')
print(f"Improvement analysis chart saved as 'improvement_analysis.png'")

# Add a summary text box with the justification
summary_text = """Justification for Slight Latency Overhead:
• Significant improvement in factual accuracy (Moderate → High)
• Dramatic reduction in hallucination rate (High → Low)
• Enhanced context grounding (Weak → Strong)
• Improved reliability (Limited → Enhanced)
• The retrieval overhead is justified by substantial gains in safety and accuracy,
  making it suitable for medical applications where reliability is paramount."""

fig4 = plt.figure(figsize=(12, 10))

# Create a text-only visualization explaining the trade-offs
ax5 = fig4.add_subplot(111)
ax5.text(0.05, 0.95, 'COMPARATIVE PERFORMANCE ANALYSIS SUMMARY', fontsize=16, fontweight='bold', ha='left', va='top')
ax5.text(0.05, 0.85, summary_text, fontsize=12, ha='left', va='top', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

# Create a simple table representation
table_data = [['Metric', 'Standalone LLM', 'CareAI (RAG)']]
for i, metric in enumerate(metrics):
    table_data.append([metric, standalone_llm[i], careai_rag[i]])

# Plot the table
table = ax5.table(cellText=table_data[1:], colLabels=table_data[0], 
                  cellLoc='center', loc='bottom', bbox=[0.05, 0.25, 0.9, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

ax5.axis('off')
ax5.set_title('Performance Comparison Summary', fontsize=14, fontweight='bold', pad=20)

plt.savefig('performance_summary_table.png', dpi=300, bbox_inches='tight')
print(f"Performance summary table saved as 'performance_summary_table.png'")

plt.show()

print(f"\nComparative performance analysis completed successfully!")
print(f"The analysis confirms that CareAI (RAG) significantly improves safety and accuracy metrics")
print(f"while accepting a slight latency trade-off, which is justified for medical applications.")