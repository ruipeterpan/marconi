# %%

"""
Sources
https://www.reddit.com/r/MachineLearning/comments/1dw6e1r/d_p_exponential_growth_of_context_length_in/
https://docs.google.com/spreadsheets/d/1xaU5Aj16mejjNvReQof0quwBJEXPOtN8nLsdBZZmepU/edit?gid=0#gid=0
https://www.artfish.ai/p/long-context-llms?utm_source=publication-search
https://x.com/get_palet/status/1798708418410344467
"""
import re
import itertools
import statistics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Data points: (x, y, label)
data_points = [
    (0, 2048, "GPT-3", 0.2),  # 2020/06
    (5.5, 8192, "GPT-4", 0),  # 2023/03
    (5.8, 100000, "Claude 1.2", -1.6),  # 2023/05
    (6.4, 16385, "GPT-3.5 Turbo", -0.3),
    (6.8, 128000, "GPT-4 Turbo", -0.3),
    (6.2, 200000, "Claude 2.1", -1.6),
    (7, 32000, "Gemini 1.0", -0.3),
    (7.5, 1000000, "Gemini 1.5", -1.6),
    (8, 2000000, "Gemini 1.5 Pro", -2.2),
    
    # (3, 8000, "GPT-3.5"),
    # (5, 32000, "Claude 1.0"),
    # (6, 100000, "Claude 2.0"),
    # (7, 400000, "GPT-4"),
    # (8, 1000000, "Claude 2.1"),
    # (9, 5000000, "Gemini Ultra 1.5")
]

# Separate data into x, y, and labels
x = [point[0] for point in data_points]
y = [point[1] for point in data_points]
labels = [point[2] for point in data_points]
text_x_offset = [point[3] for point in data_points]

# Create the figure
fig, ax = plt.subplots(figsize=(6, 4))

# plt.xticks(
#     range(0, 9), 
#     labels=["2020/06", "2020/12", "2021/06", "2021/12", "2022/06", "2022/12", "2023/06", "2023/12", "2024/06"],
#     rotation=15,
# )
ax.set_xticks(
    range(0, 9), 
    labels=["", "2021", "", "2022", "", "2023", "", "2024", ""],
    fontsize=14,
)

ax.tick_params(axis='both', which='major', labelsize=14)
# ax.set_yticks([], [], fontsize=14)
# y_ticks = [10000, 100000, 1000000]  # Desired grid lines
# y_tick_labels = ["10k", "100k", "1M"]
# plt.yticks(y_ticks, y_tick_labels, fontsize=14)

# Plot the points
ax.scatter(x, y, color='#40916C', zorder=3)

# Annotate each point with its label
for i, label in enumerate(labels):
    # Adjust text placement
    y_offset = 1.1 if i != 8 else 0.8
    ax.text(x[i] + text_x_offset[i], y[i] * y_offset, label, fontsize=11, ha='left', va='bottom', zorder=4)


# # Define lines (segments)
# x_pre = [0, 3]  # Red line (pre-context accel)
# y_pre = [2000, 8000]
# x_post = [3, 9]  # Green line (post-context accel)
# y_post = [8000, 5000000]
# # Plot the lines
# plt.plot(x_pre, y_pre, color='red', label='Pre-Context Accel')
# plt.plot(x_post, y_post, color='green', label='Post-Context Accel')

# Log scale for y-axis
ax.set_yscale('log')

# Axis labels and title
ax.set_xlabel("Model Release Date", fontsize=14)
ax.set_ylabel("Context Window Size (log scale)", fontsize=14)
# plt.title("SOTA Model Context Window Lengths Over Time")

# Add legend
# plt.legend()

# Show grid for better readability
ax.grid(which="both", linestyle="-", linewidth=0.5, alpha=0.4)
# plt.gca().yaxis.set_ticks([10000, 100000, 1000000])  # Ensure grid aligns only with these y-values


# Display the plot
plt.tight_layout()
plt.show()
fig.savefig("./context_window_over_time.pdf", dpi=500)
# fig.savefig("./context_window_over_time.pdf", dpi=500, bbox_inches='tight')

# %%
