# %%
# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

# see 1017_micro_layer_composition.txt
compositions = ("(32,4)", "(30,5)", "(28,7)", "(24,12)", "(0,36)")  # (SSM, Attn)
normalized_hitrate = {
    'Marconi': (1.0, 1.0, 1.0, 1.0, 1.0),  # 18.06, 12.3, 8.35, 5.69, 5.05
    'SGLang+': (0.62624585, 0.69186992, 0.79281437, 0.94551845, 0.9980198),  # 11.31, 8.51, 6.62, 5.38, 5.04
    'vLLM+': (0.27740864, 0.40731707, 0.6, 0.88224956, 0.9980198),  # 5.01, 5.01, 5.01, 5.02, 5.04
}
colors = {"Marconi": "#2D6A4F", "SGLang+": "#52B788", "vLLM+": "#95D5B2"}
fontsize = 14

x = np.arange(len(compositions))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(4.5, 2.7), layout="constrained")

for scheme, hitrate in normalized_hitrate.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, hitrate, width, label=scheme, color=colors[scheme])
    # ax.bar_label(rects, padding=3)
    # if scheme in ["SGLang+", "vLLM+"]:
    #     x_offset = 0.15 if scheme == "SGLang+" else 0.4
    #     for i, rate in enumerate(hitrate):
    #         diff = normalized_hitrate["Marconi"][i] / normalized_hitrate[scheme][i]
    #         ax.text(i + x_offset, rate + 0.05, f"{diff:.1f}×", rotation=90, fontsize=fontsize-3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized Hit Rate', fontsize=fontsize)
ax.set_xticks(x + width, compositions)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_xlabel("Layer Composition (SSM,Attn)", fontsize=fontsize)
ax.legend(loc="upper center", ncols=3, fontsize=fontsize, bbox_to_anchor=(0.5, 1.2), columnspacing=0.8, handlelength=0.8, frameon=False, borderaxespad=0)  # , mode="expand"
ax.set_axisbelow(True)
ax.grid(color='lightgrey', linestyle='dashed', axis="y", linewidth=0.8)

# ax.set_ylim(0, 250)

plt.show()
fig.savefig(f"../figures/eval/microbenchmark_layer_composition.pdf", dpi=500, bbox_inches='tight')
# %%
