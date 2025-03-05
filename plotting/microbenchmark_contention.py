# %%
# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

cache_sizes = ("60", "80", "100", "120", "140")
hitrate_dict = {
    'Marconi': (9.59, 22.74, 43.03, 47.92, 48.21,),
    'SGLang+': (7.72, 15.01, 25.57, 36.90, 43.89),
}
"""
Search for "sessions_per_second 5
trace_filename ./traces/swebench_sps=5_art=10_nums=100.jsonl"
in 1013_swebench, which has the first three. I probably manually ran the last two.
"""
colors = {"Marconi": "#2D6A4F", "SGLang+": "#52B788", "vLLM+": "#95D5B2"}
fontsize = 14

x = np.arange(len(cache_sizes))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(4, 2.7), layout="constrained")

for scheme, hitrate in hitrate_dict.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, hitrate, width, label=scheme, color=colors[scheme])
    # ax.bar_label(rects, padding=3)
    # if scheme in ["SGLang+", "vLLM+"]:
    #     x_offset = 0.15 if scheme == "SGLang+" else 0.4
    #     for i, rate in enumerate(hitrate):
    #         diff = hitrate_dict["Marconi"][i] / hitrate_dict[scheme][i]
    #         ax.text(i + x_offset, rate + 2, f"{diff:.1f}×", rotation=90, fontsize=fontsize-3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Token Hit Rate (%)', fontsize=fontsize)
ax.set_xticks(x + width/2, cache_sizes)
ax.set_yticks([0, 10, 20, 30, 40, 50])
# ax.set_ylim(0, 60)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_xlabel("Cache Size (GB)", fontsize=fontsize)
ax.legend(loc="upper center", ncols=3, fontsize=fontsize, bbox_to_anchor=(0.5, 1.2), columnspacing=0.8, handlelength=0.8, frameon=False, borderaxespad=0)  # , mode="expand"
ax.set_axisbelow(True)
ax.grid(color='lightgrey', linestyle='dashed', axis="y", linewidth=0.8)

# ax.set_ylim(0, 250)

plt.show()
fig.savefig(f"../figures/eval/microbenchmark_contention.pdf", dpi=500, bbox_inches='tight')
# %%
