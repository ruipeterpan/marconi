# %%
import re
import math
import pickle
import itertools
import statistics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

config = ("swebench", 100, 8e10, 10, 7.5,)
trace_name, num_sessions, capacity_bytes, sessions_per_second, avg_response_time = config
trace_filename_dir = f"../results/{capacity_bytes}/{trace_name}_sps={sessions_per_second}"
if trace_name in ["swebench"]:
    trace_filename_dir += f"_art={avg_response_time}"
# trace_filename += f"_nums={num_sessions}_finegrainedanalysis.pickle"
# NOTE to artifact evaluators:
# The exact result pickle file we used for plotting Fig. 10 was, unfortunately, still being recovered.
# It was run using an earlier version of our codebase, with a bunch of deltas 
# (e.g., in the original submission, one of our FLOP calculation functions had a typo).
# Right now, this script provides the functionality to do a fine-grained analysis on any single experiment result,
# but not exactly reproduce Fig. 10. However, we opt to retain the original version of Fig. 10
# to demonstrate the tradeoff more clearly.

trace_filename_dir += f"_nums={num_sessions}"

trace_filenames = os.listdir(trace_filename_dir)
print(f"trace_filenames {trace_filenames}")

for fname in trace_filenames:
    with open(os.path.join(trace_filename_dir, fname), "rb") as f:
        experiment_data = pickle.load(f)

    # TTFT latencies
    latency_pickle_filename = f"../data/ttft_AI21-Jamba-1.5-Mini.pickle"
    with open(latency_pickle_filename, "rb") as f:
        ttft_latencies = pickle.load(f)

    def get_approximate_ttft(seqlen):
        multiplier = 10  # during profiling, accidentally used 1e2 for s to ms conversion instead of 1e3
        # multiplier = 0.01  # display s instead of ms on x axis
        for s, ttft in ttft_latencies.items():
            if s > seqlen:
                return ttft * multiplier
        return max(ttft_latencies.values()) * multiplier

    def values_to_cdf(values):
        cdf_list = []
        values.sort()
        count = 0
        for v in values:
            count += 1
            cdf_list.append(count / len(values))
        return cdf_list

    def linear_smooth(data):
        min_value_of_data = min(data)
        # Convert dictionary to two lists for x and y
        x_vals = np.array(list(data.keys())).reshape(-1, 1)  # Reshape for linear regression
        y_vals = np.array(list(data.values()))

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(x_vals, y_vals)

        # Predict y values based on the linear model
        smoothed_data = {}
        for x in x_vals.flatten():
            y_pred = model.predict([[x]])[0]  # Predict the value
            smoothed_data[x] = y_pred
        
        min_value = min(smoothed_data.values())
        if min_value < 0:
            for k, v in smoothed_data.items():
                # smoothed_data[k] = v - min_value + 1
                smoothed_data[k] = v - min_value + 10  # assumes 100 ms for seqlen=1
                # smoothed_data[k] = v - min_value + min_value_of_data

        return smoothed_data

    ttft_latencies = linear_smooth(ttft_latencies)


    # XXX
    
    if "v2_request_history" not in experiment_data:
        continue
    
    # Token hit rate comparison
    # (cache hit or not, total tokens in request, tokens saved)
    seq_lengths = [x[1] for x in experiment_data["v1_request_history"]]
    token_hit_rates_sglang = [x[2]/x[1] for x in experiment_data["v1_request_history"]]
    token_hit_rates_marconi = [x[2]/x[1] for x in experiment_data["v2_request_history"]]
    hit_rate_diff = [100 * (x - y) for x, y in zip(token_hit_rates_marconi, token_hit_rates_sglang)]

    # fig, ax = plt.subplots(figsize=(2.5 * 1.6, 2.5))
    fig, ax = plt.subplots(figsize=(2.5 * 4, 2.5))
    fontsize = 15

    # method 1: plot actual hit rate
    # ax.scatter(seq_lengths, token_hit_rates_sglang, alpha=0.7, label="SGLang")  # , s=s
    # ax.scatter(seq_lengths, token_hit_rates_marconi, alpha=0.7, label="Marconi")
    # method 2: plot per-request hit rate difference
    ax.scatter(seq_lengths, hit_rate_diff, alpha=0.5, label="Marconi -- SGLang+", color="#40916C")  # , s=s
    ax.set_yticks([100, 50, 0, -50, -100])
    ax.ticklabel_format(axis="x", style='sci', scilimits=(3, 3))
    ax.xaxis.get_offset_text().set_fontsize(fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel("Sequence Length (# Tokens)", fontsize=fontsize)
    ax.set_ylabel("Hit Rate Diff (%)", fontsize=fontsize)
    ax.set_axisbelow(True)
    ax.legend(fontsize=fontsize)
    ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
    # ax.axhline(y=0, color="black", linestyle="dashed")

    plt.show()
    fig.savefig("../figures/eval/token_hit_rate_comparison.pdf", dpi=500, bbox_inches='tight')

    # XXX
    # Token hit rate comparison, binned
    # (cache hit or not, total tokens in request, tokens saved)
    seq_lengths = [x[1] for x in experiment_data["v1_request_history"]]
    token_hit_rates_sglang = [x[2]/x[1] for x in experiment_data["v1_request_history"]]
    token_hit_rates_marconi = [x[2]/x[1] for x in experiment_data["v2_request_history"]]
    hit_rate_diff = [100 * (x - y) for x, y in zip(token_hit_rates_marconi, token_hit_rates_sglang)]

    # fig, ax = plt.subplots(figsize=(2.5 * 1.6, 2.5))
    fig, ax = plt.subplots(figsize=(2.5 * 4, 2.5))
    colors = ['#52B788', '#40916C', '#2D6A4F', "#081C15"]
    fontsize = 15
    bar_width = 1000
    num_bins = int(32000 / bar_width)
    x = list(range(int(bar_width / 2), 32000 + bar_width, bar_width))

    data = [[] for _ in x]
    for i, seqlen in enumerate(seq_lengths):
        bin_id = math.floor(seqlen / bar_width)
        _, seqlen, v1_saved = experiment_data["v1_request_history"][i]
        _, _, v2_saved = experiment_data["v2_request_history"][i]
        data[bin_id].append((seqlen, v1_saved, v2_saved,))

    ax.axhline(y=0, color="black", linestyle="dashed", zorder=1)
    y = [100 * statistics.mean([(v2_saved - v1_saved) / seqlen for (seqlen, v1_saved, v2_saved) in bin_data]) for bin_data in data]
    ax.bar(x, y, label="Marconi − SGLang+", width=bar_width, color=colors[1], zorder=2)

    # ax.set_yticks([100, 50, 0, -50, -100])
    ax.ticklabel_format(axis="x", style='sci', scilimits=(3, 3))
    ax.xaxis.get_offset_text().set_fontsize(fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel("Sequence Length (# Tokens)", fontsize=fontsize)
    ax.set_ylabel("Avg Hit Rate Diff (%)", fontsize=fontsize)
    ax.set_axisbelow(True)
    ax.legend(fontsize=fontsize, handlelength=0.8, )
    ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)

    plt.show()
    fig.savefig("../figures/eval/token_hit_rate_comparison_bined.pdf", dpi=500, bbox_inches='tight')

    # XXX
    # TTFT CDF
    ttft_original = [get_approximate_ttft(x[1]) for x in experiment_data["v1_request_history"]]
    # ttft_vllm = [get_approximate_ttft(x[1] - x[2]) for x in experiment_data["vllm_request_history"]]
    ttft_sglang = [get_approximate_ttft(x[1] - x[2]) for x in experiment_data["v1_request_history"]]
    ttft_marconi = [get_approximate_ttft(x[1] - x[2]) for x in experiment_data["v2_request_history"]]

    # relative win
    relative_ttft_marconi = [get_approximate_ttft(x[1] - x[2]) / get_approximate_ttft(x[1]) for x in experiment_data["v2_request_history"]]
    relative_ttft_sglang = [get_approximate_ttft(x[1] - x[2]) / get_approximate_ttft(x[1]) for x in experiment_data["v1_request_history"]]
    relative_ttft_marconi = [100 * x for x in relative_ttft_marconi if x != 1.0]  # only look at the hits
    relative_ttft_sglang = [100 * x for x in relative_ttft_sglang if x != 1.0]

    print("Raw TTFT")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        sglang_p = np.percentile(ttft_sglang, p)
        marconi_p = np.percentile(ttft_marconi, p)
        winner = "Marconi" if marconi_p <= sglang_p else "LRU"
        
        marconi_improvement = 1 - marconi_p / sglang_p
        
        print(f"P{p}, {winner} better, Marconi improvement {marconi_improvement*100:.1f}%, LRU {sglang_p:.1f}, Marconi {marconi_p:.1f}")

    print("Relative TTFT")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        sglang_p = np.percentile(relative_ttft_sglang, p)
        marconi_p = np.percentile(relative_ttft_marconi, p)
        winner = "Marconi" if marconi_p <= sglang_p else "LRU"
        
        marconi_improvement = 1 - marconi_p / sglang_p
        
        print(f"P{p}, {winner} better, Marconi improvement {marconi_improvement*100:.1f}%, LRU {sglang_p:.1f}, Marconi {marconi_p:.1f}")
    print(f"Mean: SGLang {statistics.mean(relative_ttft_sglang)}, Marconi {statistics.mean(relative_ttft_marconi)}")


    fig, ax = plt.subplots(figsize=(2.5 * 4, 2.5))
    # fig, ax = plt.subplots(figsize=(2.5 * 4, 5))
    fontsize = 15
    colors = ['#52B788', '#40916C', '#2D6A4F', "#081C15"]
    linestyles = ["solid", "dotted", "dashed"]

    # axins = ax.inset_axes([0.5, 0.1, 0.45, 0.5])  # [x0, y0, width, height]: Lower-left corner of inset Axes, and its width and height.
    axins = ax.inset_axes([0.6, 0.1, 0.3, 0.5])  # [x0, y0, width, height]: Lower-left corner of inset Axes, and its width and height.

    # method 1: plot raw ttft distribution
    ax.plot(ttft_marconi, values_to_cdf(ttft_marconi), label="Marconi", color=colors[0], linestyle=linestyles[0], linewidth=2.0)
    ax.plot(ttft_sglang, values_to_cdf(ttft_sglang), label="SGLang+", color=colors[1], linestyle=linestyles[1], linewidth=2.0)
    ax.plot(ttft_original, values_to_cdf(ttft_original), label="Vanilla", color=colors[2], linestyle=linestyles[2], linewidth=2.0)

    axins.plot(ttft_marconi, values_to_cdf(ttft_marconi), label="Marconi", color=colors[0], linestyle=linestyles[0], linewidth=2.0)
    axins.plot(ttft_sglang, values_to_cdf(ttft_sglang), label="SGLang+", color=colors[1], linestyle=linestyles[1], linewidth=2.0)
    axins.plot(ttft_original, values_to_cdf(ttft_original), label="Vanilla", color=colors[2], linestyle=linestyles[2], linewidth=2.0)


    ax.set_ylabel("CDF", fontsize=fontsize)
    ax.set_xlabel("Time to First Token (ms)", fontsize=fontsize)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    # ax.set_xlabel("Time to First Token (s)", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.set_axisbelow(True)
    ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)

    x1, x2, y1, y2 = 200, 250, 0.1, 0.3  # jamba. note that the x values should actually be 10x
    # x1, x2, y1, y2 = 1000, 3000, 0.05, 0.3  # zamba
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    ax.indicate_inset_zoom(axins, edgecolor="black")


    # method 2: plot distribution of ttft improvement %
    # ax.plot(relative_ttft_marconi, values_to_cdf(relative_ttft_marconi), label="Marconi")
    # ax.plot(relative_ttft_sglang, values_to_cdf(relative_ttft_sglang), label="SGLang+")
    # ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    # ax.set_ylabel("CDF", fontsize=fontsize)
    # ax.set_xlabel("TTFT Relative to Vanilla (%)", fontsize=fontsize)
    # ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax.set_axisbelow(True)
    # ax.legend(fontsize=fontsize)
    # ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)

    plt.show()
    fig.savefig("../figures/eval/ttft_distribution.pdf", dpi=500, bbox_inches='tight')

    # %%
