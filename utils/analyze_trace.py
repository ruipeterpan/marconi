# %%
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# trace_path = "../traces/lmsys_sps=0.5_nums=100.jsonl"
# trace_path = "../traces/sharegpt_sps=0.5_nums=100.jsonl"

def get_trace_path(config):
    trace_name, num_sessions, capacity_bytes, sessions_per_second, avg_response_time = config
    trace_filename = f"../traces/{trace_name}_sps={sessions_per_second}"
    if trace_name in ["swebench"]:
        trace_filename += f"_art={avg_response_time}"
    trace_filename += f"_nums={num_sessions}.jsonl"
    return trace_filename

def analyze_and_plot_trace(trace_path):
    with open(trace_path, 'r') as json_file:
        json_list = list(json_file)

        trace = []
        for json_str in tqdm(json_list):
            result = json.loads(json_str)
            trace.append(result)
    
    print(f"Num requests: {len(trace)}")
    
    # trace = trace[:2000]
    
    fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(10, 4))
    bins = 75
    
    # input/output length distribution
    input_tokens = [x["num_input_tokens"] for x in trace]
    output_tokens = [x["num_output_tokens"] for x in trace]
    
    axs[0].hist(output_tokens, bins=bins, label="Output")
    axs[0].hist(input_tokens, bins=bins, label="Input")
    axs[0].set_xlabel("Num tokens")
    axs[0].set_ylabel("Count")
    # axs[0].set_xlim(0, 32000)
    axs[0].set_title("Input/output length distribution")
    axs[0].legend()
    
    # num of turns
    num_turns = {}
    for request in trace:
        if request["session_id"] not in num_turns:
            num_turns[request["session_id"]] = 0
        if num_turns[request["session_id"]] < request["turn_id"]:
            num_turns[request["session_id"]] = request["turn_id"]
    num_turns = num_turns.values()
    
    axs[1].hist(num_turns, bins=bins)
    axs[1].set_xlabel("Num turns/rounds")
    axs[1].set_ylabel("Count")
    # axs[1].set_xlim(0, 75)
    axs[1].set_title("Num turns/rounds")

    fig.suptitle(trace_path)
    plt.show()
    
    clean_trace_name = trace_path[:-6]  # removes .jsonl extension
    clean_trace_name = clean_trace_name[len(trace_path) - 1 - trace_path[::-1].index('/'):]
    # fig.savefig(f"../figures/trace_analysis/{clean_trace_name}.png", dpi=500)

    
configs = [  # (trace_name, num_sessions, capacity_bytes, sessions_per_second, avg_response_time)
    # ("lmsys", 100, 2e9, 1, None,),  # eff_weight: higher is better. 2 is pretty good. higher than 2 potentially helps even more
    # # ("lmsys", 100, 5e9, 1, None,),  # 0.5-0.9 is pretty good
    # # ("lmsys", 100, 1e10, 1, None,),  # smaller is good. optimal is probably somewhere 0.0-0.1
    # ("sharegpt", 100, 1e9, 5, None,),  # optimal is ~0.8
    # # ("sharegpt", 100, 2e9, 0.25, None,),  # optimal is ~0.1-0.2
    # ("swebench", 100, 1e10, 0.25, 5,),  # 0.4-1.0 is pretty good
    
    # # for getting the number of requests
    # ("lmsys", 200, 2e9, 1, None,),
    # ("lmsys", 500, 2e9, 1, None,),
    # ("swebench", 200, 1e10, 0.25, 5,),
    # ("swebench", 500, 1e10, 0.25, 5,),
    
    ("swebench", 100, 1e10, 1, 2),  # analyze new swebench trace
]

for config in configs:
    trace_path = get_trace_path(config)
    analyze_and_plot_trace(trace_path)






# %%
# %%
import json

trace_path = "/home/ubuntu/marconi/traces/mooncake_trace.jsonl"
with open(trace_path, 'r') as json_file:
    json_list = list(json_file)

trace = []
for json_str in json_list:
    result = json.loads(json_str)
    hash_ids = result["hash_ids"]
    trace.append(hash_ids)
    
    
# %%

for i, input_ids in enumerate(trace):    
    prev_trace_id = None
    for prev_id in range(i):
        prev_input_ids = trace[prev_id]
        prev_input_ids = prev_input_ids[:-1]  # if append to previous history, the last block might be a different hash, so remove it
        
        num_positions_needs_checking = min(len(input_ids), len(prev_input_ids))
        
        is_appended = all([prev_input_ids[j] == input_ids[j] for j in range(num_positions_needs_checking)])
        if is_appended:
            prev_trace_id = prev_id
    
    if prev_trace_id is not None:
        print("="*50)
        print(f"Trace {i}, previous trace {prev_trace_id}")
        print(f"Previous trace: {trace[prev_trace_id]}")
        print(f"Current trace: {input_ids}")
        

# %%
num_token_blocks = 0
num_blocks_required = 0  # assuming perfect apc, only need to prefill this many blocks
for i, input_ids in enumerate(trace):
    num_token_blocks += len(input_ids)
    
    max_reusable_blocks = 0
    for prev_id in range(i):
        prev_input_ids = trace[prev_id]
        num_positions_needs_checking = min(len(input_ids), len(prev_input_ids))
        token_block_match = [prev_input_ids[j] == input_ids[j] for j in range(num_positions_needs_checking)]
        if False in token_block_match:
            token_block_match = len(token_block_match[:token_block_match.index(False)])
        else:
            token_block_match = len(token_block_match)
        
        if token_block_match > max_reusable_blocks:
            max_reusable_blocks = token_block_match
    
    print(f"Request {i}, can reuse {max_reusable_blocks} blocks")
    num_blocks_required += (len(input_ids) - max_reusable_blocks)
    
    if i == 500:
        break
    
print(f"num_token_blocks {num_token_blocks}, num_blocks_required {num_blocks_required}")
print(f"Theoretical throughput savings: {100 - 100 * (num_blocks_required / num_token_blocks):.2f}%")
print(f"Theoretical throughput improvement: {num_token_blocks / num_blocks_required:.4f}X")
    

# %%
# test insertion into radix tree
from radix_cache_marconi import RadixCache

radix_tree = RadixCache()

for i, input_ids in enumerate(trace):
    print("="*50)
    radix_tree.insert(
        input_ids,
        state_at_leaf=2*i,
        state_at_branchoff=2*i+1,
    )
    radix_tree.pretty_print()
    
    if i == 100:
        break




# %%
import sys
sys.path.insert(0, "..") 
from utils import mooncake_trace

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
from matplotlib.ticker import PercentFormatter

input_len_distribution = [x["input_length"] for x in mooncake_trace[:500]]
fig, ax = plt.subplots(layout='constrained', figsize=(5, 4))

ax.hist(input_len_distribution, bins=50)
ax.set_xlim(0, 60000)
ax.set_xlabel("Seq len (# tokens)")
plt.show()