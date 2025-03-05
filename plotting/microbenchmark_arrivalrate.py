# %%
# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

art = (5.0, 7.5, 10.0)
hitrate_dict = {  # first version -- unsure where these numbers are from
    'Marconi': (25.90, 24.25, 24.08),
    'SGLang+': (18.83, 16.60, 15.49),
}
# hitrate_dict = {  # 1013_swebench_nums=100_wind=200. capacity_bytes 100000000000.0, sessions_per_second 5
#     'Marconi': (39.63, 41.61, 43.03),
#     'SGLang+': (26.10, 25.11, 25.57),
# }
colors = {"Marconi": "#2D6A4F", "SGLang+": "#52B788", "vLLM+": "#95D5B2"}
fontsize = 14

x = np.arange(len(art))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(4, 2.7), layout="constrained")

for scheme, hitrate in hitrate_dict.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, hitrate, width, label=scheme, color=colors[scheme])
    # ax.bar_label(rects, padding=3)
    if scheme in ["SGLang+", "vLLM+"]:
        x_offset = 0.175 if scheme == "SGLang+" else 0.4
        for i, rate in enumerate(hitrate):
            diff = hitrate_dict["Marconi"][i] / hitrate_dict[scheme][i]
            ax.text(i + x_offset, rate + 2, f"{diff:.1f}×", rotation=90, fontsize=fontsize-2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Token Hit Rate (%)', fontsize=fontsize)
ax.set_xticks(x + width/2, art)
# ax.set_ylim(0, 60)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_xlabel("Avg Response Time (s)", fontsize=fontsize)
ax.legend(loc="upper center", ncols=3, fontsize=fontsize, bbox_to_anchor=(0.5, 1.2), columnspacing=0.8, handlelength=0.8, frameon=False, borderaxespad=0)  # , mode="expand"
ax.set_axisbelow(True)
ax.grid(color='lightgrey', linestyle='dashed', axis="y", linewidth=0.8)

# ax.set_ylim(0, 250)

plt.show()
fig.savefig(f"../figures/eval/microbenchmark_art.pdf", dpi=500, bbox_inches='tight')
# %%
import matplotlib.pyplot as plt
import numpy as np

sps = (0.5, 1, 2)
hitrate_dict = {  # # 1013_swebench, art=7.5_nums=100.jsonl, capacity_bytes 100000000000.0
    'Marconi': (48.70, 44.92, 43.02),
    'SGLang+': (35.18, 29.27, 26.36),
}

colors = {"Marconi": "#2D6A4F", "SGLang+": "#52B788", "vLLM+": "#95D5B2"}
fontsize = 14

x = np.arange(len(sps))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(4, 2.7), layout="constrained")

for scheme, hitrate in hitrate_dict.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, hitrate, width, label=scheme, color=colors[scheme])
    # ax.bar_label(rects, padding=3)
    if scheme in ["SGLang+", "vLLM+"]:
        x_offset = 0.175 if scheme == "SGLang+" else 0.4
        for i, rate in enumerate(hitrate):
            diff = hitrate_dict["Marconi"][i] / hitrate_dict[scheme][i]
            ax.text(i + x_offset, rate + 3, f"{diff:.1f}×", rotation=90, fontsize=fontsize-2)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Token Hit Rate (%)', fontsize=fontsize)
ax.set_xticks(x + width/2, sps)
# ax.set_ylim(0, 60)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_xlabel("Num Sessions per Second", fontsize=fontsize)
ax.legend(loc="upper center", ncols=3, fontsize=fontsize, bbox_to_anchor=(0.5, 1.2), columnspacing=0.8, handlelength=0.8, frameon=False, borderaxespad=0)  # , mode="expand"
ax.set_axisbelow(True)
ax.grid(color='lightgrey', linestyle='dashed', axis="y", linewidth=0.8)

# ax.set_ylim(0, 250)

plt.show()
# fig.savefig(f"../figures/eval/microbenchmark_sps.pdf", dpi=500, bbox_inches='tight')
# %%
