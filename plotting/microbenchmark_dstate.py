# %%
# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

dstates = ("128", "64", "32", "16")
normalized_hitrate = {
    'Marconi': (1.0, 1.0, 1.0, 1.0),  # 31.55, 29.88, 30.19, 29.36
    'SGLang+': (0.51949287, 0.58199465, 0.59622392, 0.6256812),  # 16.39, 17.39, 18.00, 18.37
    'vLLM+': (0.02820919, 0.0502008, 0.10400795, 0.17608992),  # 0.89, 1.5, 3.14, 5.17
}
colors = {"Marconi": "#2D6A4F", "SGLang+": "#52B788", "vLLM+": "#95D5B2"}
fontsize = 14

x = np.arange(len(dstates))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(4, 2.7), layout="constrained")

for scheme, hitrate in normalized_hitrate.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, hitrate, width, label=scheme, color=colors[scheme])
    # ax.bar_label(rects, padding=3)
    if scheme in ["SGLang+", "vLLM+"]:
        x_offset = 0.15 if scheme == "SGLang+" else 0.4
        for i, rate in enumerate(hitrate):
            diff = normalized_hitrate["Marconi"][i] / normalized_hitrate[scheme][i]
            ax.text(i + x_offset, rate + 0.05, f"{diff:.1f}×", rotation=90, fontsize=fontsize-3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized Hit Rate', fontsize=fontsize)
ax.set_xticks(x + width, dstates)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_xlabel("SSM State Dim", fontsize=fontsize)
ax.legend(loc="upper center", ncols=3, fontsize=fontsize, bbox_to_anchor=(0.5, 1.2), columnspacing=0.8, handlelength=0.8, frameon=False, borderaxespad=0)  # , mode="expand"
ax.set_axisbelow(True)
ax.grid(color='lightgrey', linestyle='dashed', axis="y", linewidth=0.8)

# ax.set_ylim(0, 250)

plt.show()
fig.savefig(f"../figures/eval/microbenchmark_state_dim.pdf", dpi=500, bbox_inches='tight')
# %%
