import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the retrieval evaluation results
df_retrieval = pd.read_csv('retrieval_evaluation_results.csv')

# Define the latency data
components = ['Embedding Generation', 'Pinecone Retrieval', 'LLM Generation']
times = [120, 90, 450]  # in milliseconds
total_time = 660

# Define the hallucination data
systems = ['Standalone LLM', 'CareAI (RAG)']
hallucination_rates = [28, 8]  # in percentages

# Create a comprehensive dashboard with all metrics
fig = plt.figure(figsize=(24, 18))

# Subplot 1: Retrieval Performance Metrics
ax1 = plt.subplot(3, 4, 1)
bar_width = 0.2
bars = np.arange(len(df_retrieval['k_value']))

bars1 = ax1.bar(bars - bar_width, df_retrieval['avg_precision'], bar_width, label='Precision', color='skyblue', edgecolor='navy', alpha=0.8)
bars2 = ax1.bar(bars, df_retrieval['avg_recall'], bar_width, label='Recall', color='lightcoral', edgecolor='darkred', alpha=0.8)
bars3 = ax1.bar(bars + bar_width, df_retrieval['avg_f1_score'], bar_width, label='F1-Score', color='lightgreen', edgecolor='darkgreen', alpha=0.8)

ax1.set_xlabel('K Value')
ax1.set_ylabel('Score')
ax1.set_title('Retrieval Performance Metrics', fontsize=10, fontweight='bold')
ax1.set_xticks(bars)
ax1.set_xticklabels(df_retrieval['k_value'])
ax1.grid(axis='y', alpha=0.3)
ax1.legend(fontsize=8)

# Add value labels on bars for precision, recall, and F1
for bars_set, col_name in [(bars1, 'avg_precision'), (bars2, 'avg_recall'), (bars3, 'avg_f1_score')]:
    for i, bar in enumerate(bars_set):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=6)

# Subplot 2: Latency vs K Value
ax2 = plt.subplot(3, 4, 2)
ax2.plot(df_retrieval['k_value'], df_retrieval['avg_latency_ms'], 'ro-', linewidth=2.5, markersize=8, label='Retrieval Latency', color='purple')
ax2.set_xlabel('K Value')
ax2.set_ylabel('Latency (ms)')
ax2.set_title('Retrieval Latency vs K Value', fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

# Add value labels on points
for i, v in enumerate(df_retrieval['avg_latency_ms']):
    ax2.annotate(f'{v:.1f}', xy=(df_retrieval['k_value'].iloc[i], v), 
                 xytext=(df_retrieval['k_value'].iloc[i], v + 20),
                 ha='center', va='bottom', fontsize=6, color='purple')

# Subplot 3: Response Latency Distribution (Fig. 6.2)
ax3 = plt.subplot(3, 4, 3)
bars_latency = ax3.barh(components, times, color=['skyblue', 'lightcoral', 'lightgreen'], 
                       edgecolor='navy', alpha=0.8, height=0.6)

# Add value labels on bars
for i, (bar, time) in enumerate(zip(bars_latency, times)):
    width = bar.get_width()
    ax3.text(width + 5, bar.get_y() + bar.get_height()/2, f'{time} ms', 
             ha='left', va='center', fontweight='bold', fontsize=8)

ax3.set_xlabel('Time (milliseconds)')
ax3.set_title('Fig. 6.2. Response Latency Distribution', fontsize=10, fontweight='bold')
ax3.set_xlim(0, max(times) * 1.2)

# Add total time line
ax3.axvline(x=total_time, color='red', linestyle='--', alpha=0.7, label=f'Total: {total_time} ms')
ax3.text(total_time + 10, len(components)-0.5, f'Total\n{total_time} ms', 
         ha='left', va='center', fontweight='bold', color='red', fontsize=8)

ax3.legend(fontsize=8)

# Subplot 4: Component Contribution Pie Chart
ax4 = plt.subplot(3, 4, 4)
colors = ['skyblue', 'lightcoral', 'lightgreen']
wedges, texts, autotexts = ax4.pie(times, labels=components, autopct='%1.1f%%', 
                                   colors=colors, startangle=90, pctdistance=0.85)

# Draw circle in center to create a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig.gca().add_artist(centre_circle)

# Add total time in the center
plt.text(0, 0, f'Total\n{total_time} ms', horizontalalignment='center', 
         verticalalignment='center', fontsize=8, fontweight='bold')

ax4.set_title('Response Latency Component Contributions', fontsize=10, fontweight='bold')

# Subplot 5: Hallucination Rate Comparison
ax5 = plt.subplot(3, 4, 5)
bars_hallucination = ax5.bar(systems, hallucination_rates, color=['lightcoral', 'lightgreen'], 
               edgecolor='black', alpha=0.8, width=0.6)

# Add value labels on bars
for bar, rate in zip(bars_hallucination, hallucination_rates):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{rate}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax5.set_ylabel('Hallucination Rate (%)', fontsize=10, fontweight='bold')
ax5.set_title('Hallucination Rate Comparison', fontsize=10, fontweight='bold')
ax5.set_ylim(0, max(hallucination_rates) * 1.2)
ax5.grid(axis='y', alpha=0.3)

# Add a threshold line to show acceptable hallucination rate
ax5.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Acceptable Threshold (10%)')
ax5.legend(fontsize=8)

# Subplot 6: Accurate vs Hallucinated Stacked Chart
ax6 = plt.subplot(3, 4, 6)
accurate_vals = [100 - rate for rate in hallucination_rates]
hallucinated_vals = hallucination_rates

bar_width = 0.4
bars1 = ax6.bar(systems, accurate_vals, bar_width, label='Accurate Responses', 
                color='lightblue', edgecolor='black', alpha=0.8)
bars2 = ax6.bar(systems, hallucinated_vals, bar_width, bottom=accurate_vals, 
                label='Hallucinated Responses', color='lightcoral', edgecolor='black', alpha=0.8)

ax6.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
ax6.set_title('Accurate vs Hallucinated Responses', fontsize=10, fontweight='bold')
ax6.legend(fontsize=8)
ax6.set_ylim(0, 100)

# Add percentage labels
for i, (acc, hall) in enumerate(zip(accurate_vals, hallucinated_vals)):
    # Label for accurate portion
    ax6.text(i, acc/2, f'{acc}%', ha='center', va='center', fontweight='bold', color='black', fontsize=8)
    # Label for hallucinated portion
    ax6.text(i, acc + hall/2, f'{hall}%', ha='center', va='center', fontweight='bold', color='white', fontsize=8)

# Subplot 7: Performance vs Latency Trade-off
ax7 = plt.subplot(3, 4, 7)
# Normalize values for comparison (scale to 0-1 range)
norm_precision = (df_retrieval['avg_precision'] - df_retrieval['avg_precision'].min()) / (df_retrieval['avg_precision'].max() - df_retrieval['avg_precision'].min())
norm_latency = (df_retrieval['avg_latency_ms'] - df_retrieval['avg_latency_ms'].min()) / (df_retrieval['avg_latency_ms'].max() - df_retrieval['avg_latency_ms'].min())

ax7.plot(df_retrieval['k_value'], norm_precision, 'bo-', label='Normalized Precision', linewidth=2, markersize=8)
ax7.plot(df_retrieval['k_value'], norm_latency, 'ro-', label='Normalized Latency', linewidth=2, markersize=8)

ax7.set_xlabel('K Value')
ax7.set_ylabel('Normalized Value (0-1)')
ax7.set_title('Performance vs Latency Trade-off', fontsize=10, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend(fontsize=8)

# Add value labels
for i in range(len(df_retrieval)):
    ax7.annotate(str(df_retrieval['k_value'].iloc[i]), 
                 xy=(df_retrieval['k_value'].iloc[i], norm_precision.iloc[i]),
                 xytext=(df_retrieval['k_value'].iloc[i], norm_precision.iloc[i] + 0.05),
                 ha='center', va='bottom', fontsize=6)

# Subplot 8: Optimal K Recommendation
ax8 = plt.subplot(3, 4, 8)
# Find the optimal k based on F1 score
optimal_idx = df_retrieval['avg_f1_score'].idxmax()
optimal_k = df_retrieval.loc[optimal_idx, 'k_value']
optimal_precision = df_retrieval.loc[optimal_idx, 'avg_precision']
optimal_recall = df_retrieval.loc[optimal_idx, 'avg_recall']
optimal_f1 = df_retrieval.loc[optimal_idx, 'avg_f1_score']
optimal_latency = df_retrieval.loc[optimal_idx, 'avg_latency_ms']

metrics = ['Precision', 'Recall', 'F1-Score']
values = [optimal_precision, optimal_recall, optimal_f1]

# Create a bar chart for optimal k performance
bars_optimal = ax8.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'], 
                      edgecolor='navy', alpha=0.8)

# Add value labels on bars
for bar, value in zip(bars_optimal, values):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.2f}', ha='center', va='bottom', fontsize=8)

ax8.set_ylabel('Score')
ax8.set_title(f'Optimal K={optimal_k} Performance Profile\nLatency: {optimal_latency:.1f}ms', fontsize=10, fontweight='bold')
ax8.grid(axis='y', alpha=0.3)

# Add summary text box
summary_text = f"""Optimal K Value: {optimal_k}
Precision: {optimal_precision:.3f}
Recall: {optimal_recall:.3f}
F1-Score: {optimal_f1:.3f}
Latency: {optimal_latency:.1f} ms"""

ax8.text(0.02, 0.98, summary_text, transform=ax8.transAxes, fontsize=7,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Subplot 9: Hallucination Reduction Summary
ax9 = plt.subplot(3, 4, 9)
reduction_rate = ((hallucination_rates[0] - hallucination_rates[1]) / hallucination_rates[0]) * 100
absolute_reduction = hallucination_rates[0] - hallucination_rates[1]

# Create a pie chart showing the reduction
labels = ['Reduced\nHallucinations', 'Remaining\nHallucinations']
sizes = [reduction_rate, 100-reduction_rate]
colors = ['#2E8B57', '#FF6347']  # Green for reduction, red for remaining

wedges, texts, autotexts = ax9.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                   colors=colors, startangle=90)

ax9.set_title(f'Hallucination Reduction\n({reduction_rate:.1f}% reduction)', fontsize=10, fontweight='bold')

# Add summary text
summary_hallucination = f"""Standalone LLM: {hallucination_rates[0]}%
CareAI (RAG): {hallucination_rates[1]}%
Reduction: {absolute_reduction} pp"""

ax9.text(0, 0, summary_hallucination, horizontalalignment='center', 
         verticalalignment='center', fontsize=9, fontweight='bold')

# Subplot 10: Overall System Benefits
ax10 = plt.subplot(3, 4, 10)
benefits_data = ['71.4% Less\nHallucinations', f'{optimal_k}-way\nRetrieval', f'<{int(total_time/1000)}s\nResponse Time', 'Context-\nAware']
benefits_colors = ['lightgreen', 'lightblue', 'lightskyblue', 'lightcoral']

bars_benefits = ax10.bar(range(len(benefits_data)), [1]*len(benefits_data), 
                         color=benefits_colors, edgecolor='black', alpha=0.8)

for i, (bar, label) in enumerate(zip(bars_benefits, benefits_data)):
    ax10.text(bar.get_x() + bar.get_width()/2., 0.5, label, 
              ha='center', va='center', fontweight='bold', fontsize=9)

ax10.set_yticks([])
ax10.set_xticks(range(len(benefits_data)))
ax10.set_xticklabels(['Reliability', 'Accuracy', 'Speed', 'Context'], rotation=45, ha='right')
ax10.set_title('Overall System Benefits', fontsize=10, fontweight='bold')

# Subplot 11: Performance Summary Radar
ax11 = plt.subplot(3, 4, 11, projection='polar')

# Data for radar chart - normalized values
categories = ['Precision', 'Recall', 'F1-Score', 'Latency\n(Normalized)', 'Low\nHallucination']
values = [optimal_precision, optimal_recall, optimal_f1, 
          1 - (optimal_latency / df_retrieval['avg_latency_ms'].max()),  # Lower latency is better
          1 - (hallucination_rates[1] / 100)]  # Lower hallucination rate is better

# Normalize all values to 0-1 scale
max_vals = [1.0, 1.0, 1.0, 1.0, 1.0]  # Max possible values for each metric
values_normalized = [v/max_v for v, max_v in zip(values, max_vals)]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
values_normalized += values_normalized[:1]  # Complete the circle
angles += angles[:1]

ax11.fill(angles, values_normalized, alpha=0.25, color='blue')
ax11.plot(angles, values_normalized, color='blue', linewidth=2)

ax11.set_xticks(angles[:-1])
ax11.set_xticklabels(categories)
ax11.set_ylim(0, 1)
ax11.set_title('System Performance\nRadar', fontsize=10, fontweight='bold', pad=20)

# Subplot 12: Key Takeaways
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')

takeaways = [
    f"✓ {reduction_rate:.1f}% reduction in hallucinations vs standalone LLM",
    f"✓ Optimal K={optimal_k} provides best performance-latency balance",
    f"✓ Total response time: {total_time}ms",
    f"✓ Precision: {optimal_f1:.3f}, Recall: {optimal_recall:.3f}",
    f"✓ Medical knowledge grounded in retrieved context",
    f"✓ Suitable for healthcare applications"
]

for i, takeaway in enumerate(takeaways):
    ax12.text(0.1, 0.9 - i*0.12, takeaway, fontsize=10, fontweight='bold', 
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))

ax12.set_title('Key Takeaways', fontsize=12, fontweight='bold')

# Adjust layout
plt.tight_layout(pad=3.0)

# Add a main title
fig.suptitle('Medical Chatbot: Complete Performance & Reliability Analysis Dashboard', 
             fontsize=18, fontweight='bold', y=0.98)

# Save the comprehensive figure
plt.savefig('complete_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

print("Complete analysis dashboard saved as 'complete_analysis_dashboard.png'")
print(f"\nComplete analysis with hallucination reduction metrics completed successfully!")
print(f"CareAI reduces hallucinations by {reduction_rate:.1f}% compared to standalone LLM.")
print(f"Optimal configuration identified: K={optimal_k} with {optimal_f1:.3f} F1-score and {total_time}ms response time.")