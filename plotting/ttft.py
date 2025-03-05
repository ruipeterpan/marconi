# %%
import re
import pickle
import itertools
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


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

# TTFT latencies
latency_pickle_filename = f"../data/ttft_AI21-Jamba-1.5-Mini.pickle"
with open(latency_pickle_filename, "rb") as f:
    ttft_latencies = pickle.load(f)
ttft_latencies = linear_smooth(ttft_latencies)



def analyze_ttft(log_filename):
    log_data = ""
    with open(log_filename, 'r') as f:
        log_data += f.read()

    schemes_to_evaluate = ["vLLM+", "V2", "V3"]
    
    log_entries = [entry for entry in log_data.split('==================================================') if entry.strip()]
    
    capacity_bytes_pattern = re.compile(r"capacity_bytes (?P<capacity>[\d\.e]+)")
    
    vllm_win_list = []
    sglang_win_list = []
    marconi_win_list = []
    
    marconi_win_over_sglang_list = []
    marconi_win_over_vllm_list = []
    
    raw_ttft_list_vanilla = []
    raw_ttft_list_vllm = []
    raw_ttft_list_sglang = []
    raw_ttft_list_marconi = []
    
    raw_reduction_vllm_list = []
    raw_reduction_sglang_list = []
    raw_reduction_marconi_list = []
    
    for entry in log_entries:
        
        if "swebench" in log_filename and "capacity_bytes 20000000000.0" in entry:
            continue
        
        match = capacity_bytes_pattern.search(entry)
        if match:
            capacity_bytes = float(match.group('capacity'))
            # print(capacity_bytes, type(capacity_bytes))
        
        pickle_filename = "." + entry[entry.find("stored in ")+10:entry.rfind("\n")]
        
        ttft_original, ttft_vllm, ttft_sglang, ttft_marconi, \
            raw_reduction_vllm, raw_reduction_sglang, raw_reduction_marconi \
                = get_ttft(pickle_filename)
                
        raw_ttft_list_vanilla.append(ttft_original)
        raw_ttft_list_vllm.append(ttft_vllm)
        raw_ttft_list_sglang.append(ttft_sglang)
        raw_ttft_list_marconi.append(ttft_marconi)
        
        # % win
        # vllm_win = (ttft_original - ttft_vllm) / ttft_original
        # sglang_win = (ttft_original - ttft_sglang) / ttft_original
        # marconi_win = (ttft_original - ttft_marconi) / ttft_original
        
        vllm_win = ttft_vllm / ttft_original
        sglang_win = ttft_sglang / ttft_original
        marconi_win = ttft_marconi / ttft_original
        
        # # compare to baselines. e.g. vllm saves 0.5% in p95 ttft, marconi saves 50%. The diff is 49.5%
        marconi_win_over_vllm = vllm_win - marconi_win
        marconi_win_over_sglang = sglang_win - marconi_win
        # # here, the difference is 100x. difficult to use: sometimes there are infs due to baselines not having any savings...
        # marconi_win_over_vllm = (1 - marconi_win) / (1 - vllm_win)
        # marconi_win_over_sglang = (1 - sglang_win) / (1 - vllm_win)
        marconi_win_over_vllm_list.append(marconi_win_over_vllm)
        marconi_win_over_sglang_list.append(marconi_win_over_sglang)
        
        vllm_win_list.append(vllm_win)
        sglang_win_list.append(sglang_win)
        marconi_win_list.append(marconi_win)
        
        raw_reduction_vllm_list.append(raw_reduction_vllm)
        raw_reduction_sglang_list.append(raw_reduction_sglang)
        raw_reduction_marconi_list.append(raw_reduction_marconi)
        
        # print(f"filename {pickle_filename}, ttft_original {ttft_original}, ttft_vllm {ttft_vllm}, ttft_sglang {ttft_sglang}, ttft_marconi {ttft_marconi}, ")
    print('='*50)
    # print(f"vllm_win_list {vllm_win_list}")
    # print(f"sglang_win_list {sglang_win_list}")
    # print(f"marconi_win_list {marconi_win_list}")
    
    print("marconi's biggest reduction compared to vanilla inference")
    print(f"1 - min(marconi_win_list) {1 - min(marconi_win_list)}")
    
    print("raw reduction in ms compared to vanilla inference")
    print(f"max(raw_reduction_vllm_list) {max(raw_reduction_vllm_list)}")
    print(f"max(raw_reduction_sglang_list) {max(raw_reduction_sglang_list)}")
    print(f"max(raw_reduction_marconi_list) {max(raw_reduction_marconi_list)}")
    
    print(f"percentage reduction compared to baselines")
    #  e.g. vllm saves 0.5% in p95 ttft, marconi saves 50%. The diff is 49.5%, not 100x.
    # print(f"marconi_win_over_vllm_list {marconi_win_over_vllm_list}")
    # print(f"marconi_win_over_sglang_list {marconi_win_over_sglang_list}")
    print(f"max(marconi_win_over_vllm_list) {max(marconi_win_over_vllm_list)}")
    print(f"max(marconi_win_over_sglang_list) {max(marconi_win_over_sglang_list)}")
    
    print(f"raw reduction in ms compared to vllm and sglang in the experiment with the biggest % reduction")
    expr_id = marconi_win_over_vllm_list.index(max(marconi_win_over_vllm_list))
    raw_reduction_over_vllm = raw_ttft_list_vllm[expr_id] - raw_ttft_list_marconi[expr_id]
    print(f"raw_reduction_over_vllm {raw_reduction_over_vllm}ms")
    expr_id = marconi_win_over_sglang_list.index(max(marconi_win_over_sglang_list))
    raw_reduction_over_sglang = raw_ttft_list_sglang[expr_id] - raw_ttft_list_marconi[expr_id]
    print(f"raw_reduction_over_sglang {raw_reduction_over_sglang}ms")
    

    
    return vllm_win_list, sglang_win_list, marconi_win_list


def get_ttft(pickle_filename):
    with open(pickle_filename, "rb") as f:
        p = pickle.load(f)
    
    vllm_request_history = p["vllm_request_history"]
    v1_request_history = p["v1_request_history"]
    v2_request_history = p["v2_request_history"]
    
    # raw ttft at a percentile.
    ttft_original, ttft_vllm = get_ttft_from_request_history(vllm_request_history)
    _, ttft_sglang = get_ttft_from_request_history(v1_request_history)
    _, ttft_marconi = get_ttft_from_request_history(v2_request_history)
    
    # raw difference (ms) of ttft at that percentile
    raw_reduction_vllm = ttft_original - ttft_vllm
    raw_reduction_sglang = ttft_original - ttft_sglang
    raw_reduction_marconi = ttft_original - ttft_marconi
    
    return ttft_original, ttft_vllm, ttft_sglang, ttft_marconi, raw_reduction_vllm, raw_reduction_sglang, raw_reduction_marconi
    
    
def get_ttft_from_request_history(request_history):
    ttft_original = [get_approximate_ttft(x[1]) for x in request_history]
    ttft_scheme = [get_approximate_ttft(x[1] - x[2]) for x in request_history]
    
    # total time spent on prefill
    # return sum(ttft_original), sum(ttft_scheme)
    # p50 ttft in each trace
    return np.percentile(ttft_original, 95), np.percentile(ttft_scheme, 95)
    

# %%
fig, axs = plt.subplots(1, 3, figsize=(6.5, 1.7), sharey=True)  # layout='constrained', 
# colors = ["#2D6A4F", "#52B788", "#95D5B2"]
colors = ['#52B788', '#40916C', '#2D6A4F', "#081C15"]
linestyles = ["solid", "dotted", "dashed"]
fontsize = 14

for fig_id, log_filename in enumerate([
    # '../logs/1011_lmsys_nums=100_wind=1000_initw=0.txt',
    # '../logs/1010_sharegpt.txt',  # window is either 200, 500, or 1000. Prob 200?
    # '../logs/1010_swebench.txt'  # window 200, init weight 1
    # "../logs/1013_swebench_nums=100_wind=200.txt",
    
    # SOSP version
    "../logs/1029_lmsys_initw0.0_wind=1000.txt",
    "../logs/1029_sharegpt_initw0.0_wind=5x.txt",
    "../logs/1029_swebench_initw0.0_wind=5x.txt",
]):
    vllm_win_list, sglang_win_list, marconi_win_list = analyze_ttft(log_filename)
    
    ax = axs[fig_id]
    
    ax.plot(marconi_win_list, values_to_cdf(marconi_win_list), label="Marconi", linestyle=linestyles[0], color=colors[0])
    ax.plot(sglang_win_list, values_to_cdf(sglang_win_list), label="SGLang+", linestyle=linestyles[1], color=colors[1])
    ax.plot(vllm_win_list, values_to_cdf(vllm_win_list), label="vLLM+", linestyle=linestyles[2], color=colors[2])  
    
    # xlim_list = [
    #     (0.5, 1.0),
    #     (0.2, 1.0),
    #     (0.4, 1.0),
    # ]
    # ax.set_xlim(xlim_list[fig_id])
    xticks_list = [
        [0.7, 0.8, 0.9, 1.0],
        [0.25, 0.5, 0.75, 1.0],
        # [0.5, 0.75, 1.0],
        [0.6, 0.8, 1.0],
    ]
    ax.set_xticks(xticks_list[fig_id])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    if fig_id == 0:
        ax.set_ylabel("CDF", fontsize=fontsize)
    # ax.set_xlabel("Total TTFT Relative to Vanilla")
    title = [
        "(a) LMSys", "(b) ShareGPT", "(c) SWEBench"
    ]
    ax.set_xlabel(title[fig_id], y=-0.6, fontsize=fontsize)
    ax.set_axisbelow(True)  # puts the grid below the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
    if fig_id == 1:
        ax.legend(loc="upper center", ncols=3, fontsize=fontsize, bbox_to_anchor=(0.5, 1.25), handlelength=1.5, frameon=False, borderaxespad=0)  # , mode="expand"
        
# title = "Total TTFT Relative to No Prefix Caching"
title = "P95 TTFT Relative to Vanilla Inference"
fig.text(0.5, -0.2, title, ha='center', va='top', fontsize=fontsize)
plt.show()
# fig.savefig("../figures/eval/ttft_total_distribution.pdf", dpi=500, bbox_inches='tight')
# fig.savefig("../figures/eval/ttft_distribution_p95.pdf", dpi=500, bbox_inches='tight')

# %%
