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

def plot_sglang_comparison(log_filename_list, colors=None):
    # log_filename: a list of strings.
    
    data = []  # list of lists of wins over sglang+. one for each of: lmsys, sharegpt, swebench.
    
    for filename in log_filename_list:
        log_data = ""
        with open(filename, 'r') as f:
            log_data += f.read()
    
        schemes_to_evaluate = ["V2"]
        
        # Split log data into separate entries using the '====' separator
        log_entries = [entry for entry in log_data.split('==================================================') if entry.strip()]
        
        pattern = re.compile(r'(?P<scheme>\w+[\+]?) compared to V1: hit_rate_win (?P<hit_rate_win>-?[\d.]+)%, flops_saved_win (?P<flops_saved_win>-?[\d.]+)%')
        
        # extract metrics from each log entry
        d_keys = list(itertools.chain.from_iterable(
            # [[f'{scheme}_hit_rate_win', f'{scheme}_flops_saved_win'] for scheme in schemes_to_evaluate]
            [[f'{scheme}_hit_rate_win'] for scheme in schemes_to_evaluate]
        ))
        results = {k: [] for k in d_keys}
        
        for entry in log_entries:
            # Find all comparison entries
            for match in re.finditer(pattern, entry):
                scheme = match.group('scheme')
                hit_rate_win = float(match.group('hit_rate_win'))
                # flops_saved_win = float(match.group('flops_saved_win'))
                
                if scheme in [str(i) for i in schemes_to_evaluate]:
                    results[f'{scheme}_hit_rate_win'].append(hit_rate_win)
                    # results[f'{scheme}_flops_saved_win'].append(flops_saved_win)
        data.append(results[f'V2_hit_rate_win'])

    print(f"data len: {[len(x) for x in data]}")
    print(f"Max improvement (%): {[max(x) for x in data]}")
    print(f"P95 improvement (%): {[np.percentile(x, 95) for x in data]}")
    print(f"P90 improvement (%): {[np.percentile(x, 90) for x in data]}")
    print(f"P50 improvement (%): {[np.percentile(x, 50) for x in data]}")
    print(f"P10 improvement (%): {[np.percentile(x, 10) for x in data]}")
    print(f"P5 improvement (%): {[np.percentile(x, 5) for x in data]}")
    print(f"Min improvement (%): {[min(x) for x in data]}")

    fig, ax = plt.subplots(figsize=(5, 2.5))

    # custom_xlabels = ["LMSys", "ShareGPT", "SWEBench"]
    custom_xlabels = ["SWEBench", "ShareGPT", "LMSys"]

    # If colors not provided, use a default color list
    if colors is None:
        colors = ['#52B788', '#40916C', '#2D6A4F', "#081C15"]
    linewidth = 1.5
    fontsize = 12
    box_width = 0.5

    # Create boxplot without setting the box color initially
    data = [data[2], data[1], data[0]]  # swap the order of the ylabels
    bp = ax.boxplot(
        data,
        vert=False, widths=box_width,  # create a horizontal boxplot
        showfliers=False,
        whiskerprops=dict(linewidth=1.5),  # set whisker properties
        whis=[5, 95],
    )
    
    # Manually set colors for each box and its corresponding whiskers and median
    for i, box in enumerate(bp['boxes']):
        box.set_color(colors[i])  # Set the color of the box
        box.set_linewidth(linewidth)  # Set the linewidth of the box
        # Set the color of the corresponding whiskers and medians
        # Set the color and linewidth of the corresponding whiskers, medians, and caps
        bp['whiskers'][2*i].set_color(colors[i])  # Lower whisker
        bp['whiskers'][2*i].set_linewidth(linewidth)  # Lower whisker linewidth
        bp['whiskers'][2*i + 1].set_color(colors[i])  # Upper whisker
        bp['whiskers'][2*i + 1].set_linewidth(linewidth)  # Upper whisker linewidth
        bp['medians'][i].set_color(colors[i])  # Median line
        bp['medians'][i].set_linewidth(linewidth)  # Median linewidth
        bp['caps'][2*i].set_color(colors[i])  # Lower cap
        bp['caps'][2*i].set_linewidth(linewidth)  # Lower cap linewidth
        bp['caps'][2*i + 1].set_color(colors[i])  # Upper cap
        bp['caps'][2*i + 1].set_linewidth(linewidth)  # Upper cap linewidth
        
        # ax.scatter(i + 1, statistics.mean(data[i]), color=colors[i], marker='.', s=100, label='Mean')  # vertical
        ax.scatter(statistics.mean(data[i]), i + 1, color=colors[i], marker='.', s=100, label='Mean')  # horizontal

    # # vertical
    # # Set custom x-axis labels and scatter plot for means
    # ax.tick_params(axis='y', which='major', labelsize=fontsize)
    # ax.set_xticklabels(custom_xlabels, rotation=0, fontsize=fontsize)
    # ax.set_ylabel('Token Hit Rate Win:\nMarconi over SGLang+ (%)', fontsize=fontsize)
    # ax.grid(color='lightgrey', linestyle='dashed', axis="y", linewidth=0.8)
    # # ax.set_yticks([0, 25, 50, 75, 100, 125, 150])
    
    # Set custom x-axis labels and scatter plot for means
    ax.tick_params(axis='x', which='major', labelsize=fontsize)
    ax.set_yticklabels(custom_xlabels, rotation=0, fontsize=fontsize)
    ax.set_xlabel('Marconi’s Token Hit Rate Improvement over SGLang+ (%)', fontsize=fontsize)
    ax.xaxis.set_label_coords(0.375, -0.2)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200])  # Define all tick locations
    ax.set_xticklabels(['0', '', '50', '', '100', '', '150', '', '200'])  # Define labels, leave blanks for hidden labels
    ax.grid(color='lightgrey', linestyle='dashed', axis="x", linewidth=0.8)

    # Display the plots
    plt.tight_layout()
    plt.show()
    fig.savefig(f"../figures/eval/sglang_comparison_horizontal.pdf", dpi=500, bbox_inches='tight')


# %%
plot_sglang_comparison([
    "../logs/lmsys.txt",
    "../logs/sharegpt.txt",
    "../logs/swebench.txt",
])


# %%
