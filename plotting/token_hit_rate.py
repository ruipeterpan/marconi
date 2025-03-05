# %%
import re
import itertools
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_token_hit_rate(log_filename, colors=None):
    # log_filename: a string of the path of the log
    
    log_data = ""
    with open(log_filename, 'r') as f:
        log_data += f.read()

    schemes_to_evaluate = ["vLLM+", "V2"]
    
    # Split log data into separate entries using the '====' separator
    log_entries = [entry for entry in log_data.split('==================================================') if entry.strip()]
    
    pattern = re.compile(r"(?P<scheme>\w+[\+]?): hit rate (?P<hit_rate>[\d\.]+)%, FLOPs saved (?P<flops_saved>[\d\.e]+)")
    
    # extract metrics from each log entry
    d_keys = list(itertools.chain.from_iterable(
        [[f'{scheme}_hit_rate'] for scheme in schemes_to_evaluate]
    ))
    results = {k: [] for k in d_keys}
    
    for entry in log_entries:
        # Find all comparison entries
        for match in re.finditer(pattern, entry):
            scheme = match.group('scheme')
            hit_rate = float(match.group('hit_rate'))
            
            if scheme in [str(i) for i in schemes_to_evaluate]:
                results[f'{scheme}_hit_rate'].append(hit_rate)
    
    df = pd.DataFrame(results)
    means = df.mean()
    
    fig, ax = plt.subplots(figsize=(4, 2))

    if schemes_to_evaluate == ["vLLM+", "V1", "V2", "V3"]:
        custom_ylabels = ["vLLM+", "SGLang+", "Marconi", "Oracle"]
    elif schemes_to_evaluate == ["vLLM+", "V2"]:
        custom_ylabels = ["vLLM+", "Marconi"]
    elif schemes_to_evaluate == ["vLLM+", "V2", "V3"]:
        custom_ylabels = ["vLLM+", "Marconi", "Oracle"]

    # If colors not provided, use a default color list
    if colors is None:
        colors = ['#52B788', '#40916C', '#2D6A4F', "#081C15"]
    linewidth = 1.5
    fontsize = 12
    box_width = 0.5
    
    marconi_win_over_vllm = []
    for i in range(len(results["vLLM+_hit_rate"])):
        marconi_win_over_vllm.append(
            results["V2_hit_rate"][i] / results["vLLM+_hit_rate"][i]
        )
    
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"P{p}, Marconi improvement {np.percentile(marconi_win_over_vllm, p):.2f} times")
    print(f"Average improvement: {statistics.mean(marconi_win_over_vllm)}")
    print(f"Num experiments: {len(df.index)}")

    # Create horizontal boxplot without setting the box color initially
    bp = ax.boxplot(
        df[d_keys],
        vert=False,  # Set vert=False to create a horizontal boxplot
        showfliers=False,
        whiskerprops=dict(linewidth=1.5),  # set whisker properties
        widths=box_width,
        whis=[5, 95],
    )
    
    # Manually set colors for each box and its corresponding whiskers and median
    for i, box in enumerate(bp['boxes']):
        box.set_color(colors[i])  # Set the color of the box
        box.set_linewidth(linewidth)  # Set the linewidth of the box
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

        # Scatter plot the mean values
        ax.scatter(means[d_keys[i]], i + 1, color=colors[i], marker='.', s=100, label='Mean')

    # Set custom y-axis labels and scatter plot for means
    ax.tick_params(axis='x', which='major', labelsize=fontsize)
    ax.set_yticklabels(custom_ylabels, rotation=0, fontsize=fontsize)
    ax.set_xlabel('Token Hit Rate (%)', fontsize=fontsize)
    ax.set_axisbelow(True)
    ax.grid(color='lightgrey', linestyle='dashed', axis="x", linewidth=0.8)
    
    # Display the plots
    plt.tight_layout()
    plt.show()
    # dataset_name = log_filename.split('_', 2)[1].split('.')[0]
    dataset_name = log_filename.split('/')[-1].split('.')[0]
    fig.savefig(f"../figures/eval/{dataset_name}.pdf", dpi=500, bbox_inches='tight')


# %%
for log_filename in [
    "../logs/lmsys.txt",
    "../logs/sharegpt.txt",
    "../logs/swebench.txt",
]:
    plot_token_hit_rate(log_filename)


# %%
