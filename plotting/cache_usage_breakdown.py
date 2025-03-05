# %%
import re
import itertools
import statistics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

block_sizes = ("32", "64", "128")
reuse_rate = {
    'KVs': (24.95, 28.87, 36.28),
    'SSM States': (0.3822, 1.036, 3.26),
}
colors = {"KVs": "#2D6A4F", "SSM States": "#52B788"}  # , "#95D5B2"
fontsize = 14

x = np.arange(len(block_sizes))  # the label locations
width = 0.35  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(3, 2.7), layout="constrained")

for attribute, layer_reuse_rate in reuse_rate.items():
    offset = width * multiplier
    color = colors[attribute]
    rects = ax.bar(x + offset, layer_reuse_rate, width, label=attribute, color=color)
    # ax.bar_label(rects, padding=3)
    if attribute == "SSM States":
        for i, rate in enumerate(layer_reuse_rate):
            diff = reuse_rate["KVs"][i] / reuse_rate["SSM States"][i]
            ax.text(i + 0.225, rate + 2, f"{diff:.1f}×", rotation=90, fontsize=fontsize-2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel("Token Block Size", fontsize=fontsize)
ax.set_ylabel("% Blocks Reused", fontsize=fontsize)
ax.set_xticks(x + 0.5 * width, block_sizes, fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
# ax.legend(loc='upper left', ncols=2)
ax.legend(loc="upper center", ncols=2, fontsize=fontsize, bbox_to_anchor=(0.5, 1.2), handlelength=0.8, frameon=False, borderaxespad=0)  # , mode="expand"
# ax.set_ylim(0, 250)
ax.set_axisbelow(True)
ax.grid(color='lightgrey', linestyle='dashed', axis="y", linewidth=0.8)

plt.show()
fig.savefig(f"../figures/eval/cache_usage_breakdown.pdf", dpi=500, bbox_inches='tight')

"""
Motivational experiment
Config: ("lmsys", 100, 5e9, 0.5, None,),

block size 32: nodes_created 388799, len(token_representations) 37142, len(nodes_accessed_kv) 8682 (23.3%), len(nodes_accessed_ssm) 133 (0.35%)
token_representations: [1,2] and [1,2,3] are the same
If not: nodes_created 388799, len(token_representations) 34795, len(nodes_accessed_kv) 8682 (25.0%), len(nodes_accessed_ssm) 133 (0.38%), 65x
vLLM+: hit rate 5.56%, FLOPs saved 9106.43e12
V1: hit rate 38.55%, FLOPs saved 63030.18e12
V3: hit rate 44.54%, FLOPs saved 72809.31e12
V2: hit rate 42.03%, FLOPs saved 68801.25e12
V3 compared to V1: hit_rate_win 15.53%, flops_saved_win 15.51%
V3 compared to vLLM: hit_rate_win 700.90%, flops_saved_win 699.54%
V2 compared to V1: hit_rate_win 9.02%, flops_saved_win 9.16%
V2 compared to vLLM: hit_rate_win 655.74%, flops_saved_win 655.52%
V1 compared to vLLM: hit_rate_win 593.21%, flops_saved_win 592.15%

block size 64: nodes_created 189919, len(token_representations) 17363, len(nodes_accessed_kv) 5014 (28.9%), len(nodes_accessed_ssm) 180 (1.03%), 28x
this is not counting partial blocks into the representations

block size 128: nodes_created 91734, len(token_representations) 8639, len(nodes_accessed_kv) 3135 (36.3%), len(nodes_accessed_ssm) 282 (3.3%), 11x

block size 256: nodes_created 42645, len(token_representations) 4277, len(nodes_accessed_kv) 2275 (53%), len(nodes_accessed_ssm) 409 (9.5%)
"""
# %%
