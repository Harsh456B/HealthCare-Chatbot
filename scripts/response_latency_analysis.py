import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the latency data
components = ['Embedding Generation', 'Pinecone Retrieval', 'LLM Generation', 'Total']
times = [120, 90, 450, 660]  # in milliseconds

# Create a DataFrame for the latency data
df = pd.DataFrame({
    'Component': components,
    'Time (ms)': times
})

print("Response Latency Analysis")
print("="*50)
print(df.to_string(index=False))
print("="*50)
print(f"Average Total Response Time: {times[-1]} ms")
print(f"Fastest Component: {components[times.index(min(times[:-1]))]} ({min(times[:-1])} ms)")
print(f"Slowest Component: {components[times.index(max(times[:-1]))]} ({max(times[:-1])} ms)")

# Create Figure 6.2: Response Latency Distribution
fig, ax = plt.subplots(figsize=(12, 8))

# Create horizontal bar chart
bars = ax.barh(components[:-1], times[:-1], color=['skyblue', 'lightcoral', 'lightgreen'], 
               edgecolor='navy', alpha=0.8, height=0.6)

# Add value labels on bars
for i, (bar, time) in enumerate(zip(bars, times[:-1])):
    width = bar.get_width()
    ax.text(width + 5, bar.get_y() + bar.get_height()/2, f'{time} ms', 
            ha='left', va='center', fontweight='bold')

# Highlight the total with a different style
total_bar = ax.barh(['TOTAL'], [times[-1]], color=['gold'], 
                   edgecolor='orange', alpha=0.8, height=0.6, hatch='///')

for i, (bar, time) in enumerate(zip(total_bar, [times[-1]])):
    width = bar.get_width()
    ax.text(width + 5, bar.get_y() + bar.get_height()/2, f'{time} ms', 
            ha='left', va='center', fontweight='bold', color='darkred')

# Customize the plot
ax.set_xlabel('Time (milliseconds)', fontsize=12, fontweight='bold')
ax.set_title('Fig. 6.2. Response Latency Distribution', fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, max(times) * 1.2)  # Add some space for labels
ax.grid(axis='x', alpha=0.3)

# Add a note about the breakdown
note_text = f"""Average latency breakdown:
• Embedding Generation: {times[0]} ms ({times[0]/times[-1]*100:.1f}% of total)
• Pinecone Retrieval: {times[1]} ms ({times[1]/times[-1]*100:.1f}% of total)  
• LLM Generation: {times[2]} ms ({times[2]/times[-1]*100:.1f}% of total)
• Total: {times[3]} ms (100%)"""

# Position the note box
ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        horizontalalignment='right')

# Invert y-axis to have the fastest component at the top
ax.invert_yaxis()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('response_latency_distribution.png', dpi=300, bbox_inches='tight')
print("\nFigure 6.2 saved as 'response_latency_distribution.png'")

# Create a pie chart version to show percentage contributions
fig2, ax2 = plt.subplots(figsize=(10, 8))

# Calculate percentages for non-total components
non_total_times = times[:-1]
colors = ['skyblue', 'lightcoral', 'lightgreen']
wedges, texts, autotexts = ax2.pie(non_total_times, labels=components[:-1], autopct='%1.1f%%', 
                                   colors=colors, startangle=90, pctdistance=0.85)

# Draw circle in center to create a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig2.gca().add_artist(centre_circle)

# Add total time in the center
plt.text(0, 0, f'Total\n{times[-1]} ms', horizontalalignment='center', 
         verticalalignment='center', fontsize=16, fontweight='bold')

ax2.set_title('Response Latency Component Contributions', fontsize=14, fontweight='bold', pad=20)

# Save the pie chart
plt.savefig('latency_component_pie_chart.png', dpi=300, bbox_inches='tight')
print("Pie chart saved as 'latency_component_pie_chart.png'")

# Create a stacked bar chart showing the composition of total latency
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Create stacked bar chart
bottom = 0
for i, (component, time) in enumerate(zip(components[:-1], times[:-1])):
    ax3.bar(['Total Response Time'], time, bottom=bottom, label=component, 
            color=colors[i], edgecolor='black', alpha=0.8)
    # Add component labels inside the bars
    ax3.text(0, bottom + time/2, f'{component}\n{time} ms', 
             ha='center', va='center', fontweight='bold', color='white' if i != 1 else 'black')
    bottom += time

ax3.set_ylabel('Time (milliseconds)', fontsize=12)
ax3.set_title('Total Response Time Composition', fontsize=14, fontweight='bold', pad=20)
ax3.set_ylim(0, times[-1] * 1.1)
ax3.legend(loc='lower right')

# Add total time on top
ax3.text(0, times[-1] + 10, f'Total: {times[-1]} ms', 
         ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.savefig('latency_composition_stacked.png', dpi=300, bbox_inches='tight')
print("Stacked bar chart saved as 'latency_composition_stacked.png'")

plt.show()
print("\nAll latency analysis visualizations generated successfully!")