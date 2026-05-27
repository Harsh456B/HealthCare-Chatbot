import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the hallucination data
systems = ['Standalone LLM', 'CareAI (RAG)']
rates = [28, 8]  # in percentages

# Create a DataFrame for the data
df = pd.DataFrame({
    'System': systems,
    'Hallucination_Rate_Percent': rates
})

print("Hallucination Reduction Analysis")
print("="*40)
print(df.to_string(index=False))
print("="*40)

reduction_rate = ((rates[0] - rates[1]) / rates[0]) * 100
print(f"Hallucination Reduction Achieved: {reduction_rate:.1f}%")
print(f"CareAI reduces hallucinations by {rates[0] - rates[1]} percentage points compared to standalone LLM")

# Create the Hallucination Rate Comparison chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
bars = ax1.bar(systems, rates, color=['lightcoral', 'lightgreen'], 
               edgecolor='black', alpha=0.8, width=0.6)

# Add value labels on bars
for bar, rate in zip(bars, rates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{rate}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax1.set_ylabel('Hallucination Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Hallucination Rate Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(rates) * 1.2)
ax1.grid(axis='y', alpha=0.3)

# Add a threshold line to show acceptable hallucination rate
ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Acceptable Threshold (10%)')
ax1.legend()

# Pie charts showing the proportion of accurate vs hallucinated responses
accurate_rates = [100 - rate for rate in rates]

ax2.pie([rates[0], accurate_rates[0]], labels=['Hallucinated', 'Accurate'], 
        colors=['lightcoral', 'lightblue'], autopct='%1.1f%%', startangle=90, radius=0.8)
ax2.set_title(f'Standalone LLM:\n{rates[0]}% Hallucination', fontsize=12, fontweight='bold', pad=20)

# Add a text box with summary
summary_text = f"""Hallucination Reduction Analysis:
• Standalone LLM: {rates[0]}% hallucination rate
• CareAI (RAG): {rates[1]}% hallucination rate
• Improvement: {reduction_rate:.1f}% reduction
• Absolute reduction: {rates[0]-rates[1]} percentage points"""

fig.text(0.02, 0.02, summary_text, fontsize=10,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('hallucination_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nHallucination comparison chart saved as 'hallucination_comparison.png'")

# Create a second visualization focusing on the reduction
fig2, ax3 = plt.subplots(figsize=(10, 6))

# Create a horizontal bar chart showing the difference
y_pos = np.arange(len(systems))
bars = ax3.barh(y_pos, rates, color=['lightcoral', 'lightgreen'], 
                edgecolor='black', alpha=0.8, height=0.5)

# Add value labels
for i, (bar, rate) in enumerate(zip(bars, rates)):
    width = bar.get_width()
    ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{rate}%', 
             ha='left', va='center', fontweight='bold', fontsize=12)

ax3.set_xlabel('Hallucination Rate (%)', fontsize=12, fontweight='bold')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(systems)
ax3.set_title('Hallucination Rate Comparison - Horizontal View', fontsize=14, fontweight='bold')
ax3.set_xlim(0, max(rates) * 1.2)
ax3.grid(axis='x', alpha=0.3)

# Add reduction indicator
reduction_arrow_x = rates[0]
reduction_arrow_y = 0.2  # Position between the two bars
ax3.annotate('', xy=(rates[1], 0.2), xytext=(rates[0], 0.2),
             arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
ax3.text((rates[0] + rates[1])/2, 0.3, f'Reduction\n{rates[0]-rates[1]}pp', 
         ha='center', va='bottom', fontsize=10, fontweight='bold', color='blue')

plt.tight_layout()
plt.savefig('hallucination_reduction_detailed.png', dpi=300, bbox_inches='tight')
print(f"Detailed reduction chart saved as 'hallucination_reduction_detailed.png'")

# Create a stacked bar chart showing accurate vs hallucinated portions
fig3, ax4 = plt.subplots(figsize=(10, 6))

accurate_vals = [100 - rate for rate in rates]
hallucinated_vals = rates

bar_width = 0.5
bars1 = ax4.bar(systems, accurate_vals, bar_width, label='Accurate Responses', 
                color='lightblue', edgecolor='black', alpha=0.8)
bars2 = ax4.bar(systems, hallucinated_vals, bar_width, bottom=accurate_vals, 
                label='Hallucinated Responses', color='lightcoral', edgecolor='black', alpha=0.8)

ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax4.set_title('Accurate vs Hallucinated Responses Comparison', fontsize=14, fontweight='bold')
ax4.legend()
ax4.set_ylim(0, 100)

# Add percentage labels
for i, (acc, hall, sys) in enumerate(zip(accurate_vals, hallucinated_vals, systems)):
    # Label for accurate portion
    ax4.text(i, acc/2, f'{acc}%', ha='center', va='center', fontweight='bold', color='black')
    # Label for hallucinated portion
    ax4.text(i, acc + hall/2, f'{hall}%', ha='center', va='center', fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('accurate_vs_hallucinated_comparison.png', dpi=300, bbox_inches='tight')
print(f"Accurate vs hallucinated chart saved as 'accurate_vs_hallucinated_comparison.png'")

plt.show()

print(f"\nHallucination reduction analysis completed successfully!")
print(f"CareAI with RAG technology significantly reduces hallucinations by {reduction_rate:.1f}% compared to standalone LLM.")