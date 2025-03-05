# %%
import io
import os
import gc
import sys
import json
import pickle
import argparse
import itertools
import statistics
import numpy as np
import pandas as pd
from datetime import datetime
from pytz import timezone
from datasets import load_dataset
from tqdm import tqdm
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "..") 
from radix_cache_hybrid import RadixCache, _key_match
from radix_cache_vllm import RadixCache as RadixCacheVLLM

# My vLLM+ implementation is not great, and this avoids hitting the default recursion limit (1000)
sys.setrecursionlimit(1500)
# Automatically get number of CPUs available for parallelizing grid search
NUM_CPUS = int(os.cpu_count())
# default directory for request arrival traces
TRACE_DIR = "./traces"  
# TRACE_DIR = "/data2/ruipan/marconi/traces"

# %%
def load_request_trace(trace_path: str):
    """Loads a json request trace

    Args:
        trace_path (str): Path to the json request trace.

    Returns:
        list: List of requests. Each request is a dict with keys:
            session_id, turn_id, ts, num_input_tokens, num_output_tokens, input_tokens
    """
    with open(trace_path, 'r') as json_file:
        json_list = list(json_file)

    trace = []
    for json_str in json_list:
        result = json.loads(json_str)
        trace.append(result)
    return trace


def run_trace_with_config(
    trace_name,
    num_sessions,
    capacity_bytes,
    evict_policy_version,
    eff_weight,
    sessions_per_second,
    avg_response_time,
    num_ssm_layers,
    num_attn_layers,
    num_mlp_layers,
    d,
    n,
    bootstrap_multiplier=5,
    block_size=None,
    admission_strategy="selective",
    lru_baseline_perf=None,
    *args, **kwargs,
):
    buffer = io.StringIO()
    radix_tree_class = {
        "selective": RadixCache,
        "all": RadixCacheVLLM,
    }[admission_strategy]
    
    # Redirect stdout to the buffer
    # with redirect_stdout(buffer):
    if True:
        trace_filename = os.path.join(TRACE_DIR, f"{trace_name}_sps={sessions_per_second}")
        if trace_name in ["swebench"]:
            trace_filename += f"_art={avg_response_time}"
        trace_filename += f"_nums={num_sessions}.jsonl"
        all_requests = load_request_trace(trace_filename)
        
        if admission_strategy == "all":  # only print once for the vLLM+ run
            print(f"Running trace ({len(all_requests)} requests): {trace_filename}")
                
        radix_tree = radix_tree_class(
            capacity_bytes=capacity_bytes,
            evict_policy_version=evict_policy_version,
            eff_weight=eff_weight,
            use_logical_ts=True,
            num_ssm_layers=num_ssm_layers,
            num_attn_layers=num_attn_layers,
            num_mlp_layers=num_mlp_layers,
            d=d,
            n=n,
            block_size=block_size,
            bootstrap_multiplier=bootstrap_multiplier,
        )

        request_stats = []  # for oracle token hit rate calculation. list of tuples (num_input_tokens, max prefix len when matched with all historical requests)

        # for request_id in tqdm(range(len(all_requests))):
        for request_id in range(len(all_requests)):
            request = all_requests[request_id]
            input_tokens = request["input_tokens"]
            radix_tree.match_prefix(input_tokens)
            
            # oracle token hit rate calculation
            max_prefix_len_match = 0
            # for prev_req_id in range(request_id):
            #     prefix_len_match = _key_match(input_tokens, all_requests[prev_req_id]["input_tokens"] + all_requests[prev_req_id]["output_tokens"])
            #     if prefix_len_match > max_prefix_len_match:
            #         max_prefix_len_match = prefix_len_match
            request_stats.append((len(input_tokens), max_prefix_len_match,))
            
            output_tokens = request["output_tokens"]
            all_tokens = input_tokens + output_tokens
            # print(f"all_tokens len {len(all_tokens)}")
            radix_tree.insert(
                token_ids=all_tokens,
                state_at_leaf=request["session_id"],
                state_at_branchoff=request["session_id"],
            )
            
            # radix_tree.pretty_print()
        # print("="*50, flush=True)
        # print(f"Trace {trace_name}")
        # print(f"Theoretical optimal token hit rate: {sum([x[1] for x in request_stats]) / sum([x[0] for x in request_stats]) * 100:.2f}%", flush=True)
        # print(f"capacity_bytes {capacity_bytes}, sessions_per_second {sessions_per_second}, evict_policy_version {radix_tree.evict_policy_version}, eff_weight {radix_tree.eff_weight}, recency_weight {radix_tree.recency_weight}", flush=True)
        # radix_tree.pretty_print()
        num_cached_mamba_states, num_cached_kv_tokens = radix_tree.get_num_cached_tokens()
        # print(f"num_cached_mamba_states {num_cached_mamba_states}, num_cached_kv_tokens {num_cached_kv_tokens}", flush=True)
        request_hit_rate, token_hit_rate, total_mamba_flop_savings, total_attn_flop_savings, total_mlp_flop_savings = radix_tree.get_cache_stats(verbose=False)
        # radix_tree.get_tree_size(verbose=True)
            
            
        if lru_baseline_perf is not None:
            if token_hit_rate > lru_baseline_perf[(sessions_per_second, capacity_bytes,)][0]:
                token_hit_rate_win = token_hit_rate - lru_baseline_perf[(sessions_per_second, capacity_bytes,)][0]
                flops_savings_win = (total_mamba_flop_savings + total_attn_flop_savings + total_mlp_flop_savings) / lru_baseline_perf[(sessions_per_second, capacity_bytes,)][1] - 1
                # print(f"Better configuration! token_hit_rate_win {token_hit_rate_win * 100:.2f}%, flops_savings_win {flops_savings_win * 100:.2f}%", flush=True)
            
    # del radix_tree, all_requests, request_stats
    del all_requests, request_stats
    gc.collect()
            
    if evict_policy_version == 8:
        print(f"Finished running trace. eff_weight history: {radix_tree.eff_weight_history}")
    elif evict_policy_version == 5:
        print(f"Finished running trace. avg eff_weight: {statistics.mean([x[0] for x in radix_tree.eff_weight_history])}")
            
    return buffer.getvalue(), locals(), radix_tree, request_hit_rate, token_hit_rate, total_mamba_flop_savings, total_attn_flop_savings, total_mlp_flop_savings


def generate_configs(args):
    if args.dataset == "lmsys":
        num_sessions = [100]  # number of chat sessions
        capacity_bytes = [1e9, 2e9, 3e9, 4e9, 5e9]  # cache size in bytes
        sessions_per_second = [0.25, 0.5, 1, 2, 5, 10]  # inter-session arrival time in seconds
        return [("lmsys", n, c, s, None) for n, c, s in itertools.product(num_sessions, capacity_bytes, sessions_per_second)]
    elif args.dataset == "sharegpt":
        num_sessions = [100]
        capacity_bytes = [1e9, 2e9, 3e9, 4e9, 5e9, 1e10]
        sessions_per_second = [0.25, 0.5, 1, 2, 5, 10]
        return [("sharegpt", n, c, s, None) for n, c, s in itertools.product(num_sessions, capacity_bytes, sessions_per_second)]
    elif args.dataset == "swebench":
        num_sessions = [100]
        capacity_bytes = [4e10, 6e10, 8e10, 10e10]  # 2e10 was prob removed for mlsys
        sessions_per_second = [0.25, 0.5, 1, 2, 5, 10]
        avg_response_time = [5, 7.5, 10]  # time between request arrival within a chat session
        return [("swebench", n, c, s, a) for (n, c, s, a) in itertools.product(num_sessions, capacity_bytes, sessions_per_second, avg_response_time)]
    else:
        assert args.dataset is None, f"Running unsupported dataset {args.dataset}, please modify the code to incorporate it before running!"
        return [  # some random traces
            ("swebench", 100, 5e9, 0.25, 5,),
            ("lmsys", 100, 5e9, 0.5, None,),  # motivational experiment
        ]


# %%
parser = argparse.ArgumentParser(description="Runs the full sweep of experiments.")
parser.add_argument("--dataset", type=str, default=None, choices=['lmsys', 'sharegpt', 'swebench'],
                    help="Specifies a dataset to do sweep on. If unspecified, a single default trace will be run.")
parser.add_argument("--bootstrap_multiplier", type=int, default=5, 
                    help="Bootstrap window size multiplier")
args, _ = parser.parse_known_args()
configs = generate_configs(args)

if args.dataset == "lmsys":
    args.bootstrap_multiplier = 15

# model configs
# NVIDIA's Attention-Mamba2 Hybrid 7B model (https://arxiv.org/pdf/2406.07887)
num_ssm_layers = 24
num_attn_layers = 4
num_mlp_layers = 28
d = 4096
n = 128


for config in configs:
    trace_name, num_sessions, capacity_bytes, sessions_per_second, avg_response_time = config
    print("="*50)
    print(datetime.now(timezone('EST')).strftime('%Y-%m-%d %H:%M:%S %Z'))
    DATETIME = datetime.now(timezone('EST')).strftime("%Y_%m_%d_%H_%M_%S")  # date and time of experiment: 2024_10_14_15_55_30 -> 24/10/14, 3:55:30P…
    print(f"Cache size {capacity_bytes/1e9:.1f} GB, sessions per second: {sessions_per_second}")
    
    # run vLLM+
    configs = {
        "trace_name": trace_name,
        "num_sessions": num_sessions,
        "capacity_bytes": capacity_bytes,
        "evict_policy_version": 1,  # doesn't matter when invoking vLLM+
        "eff_weight": 0.0,  # doesn't matter when invoking vLLM+
        "sessions_per_second": sessions_per_second,
        "avg_response_time": avg_response_time,
        "admission_strategy": "all",
        "num_ssm_layers": num_ssm_layers,
        "num_attn_layers": num_attn_layers,
        "num_mlp_layers": num_mlp_layers,
        "d": d,
        "n": n,
        "block_size": 32,
    }
    output, local_args, radix_tree, request_hit_rate, token_hit_rate, total_mamba_flop_savings, total_attn_flop_savings, total_mlp_flop_savings = \
        run_trace_with_config(**configs)
    total_flops_savings = total_mamba_flop_savings + total_attn_flop_savings + total_mlp_flop_savings
    locals()[f"vllm_max_hit_rate"], locals()[f"vllm_max_flops_saved"] = token_hit_rate, total_flops_savings
    locals()[f"vllm_request_history"] = radix_tree.request_history
    print(f"vLLM+: hit rate {token_hit_rate*100:.2f}%, FLOPs saved {total_flops_savings/1e12:.2f}e12")
    
    # print(f"Motivational experiment:")
    # nodes_created = radix_tree.nodes_created
    # token_representations = radix_tree.token_representations
    # nodes_accessed_kv = radix_tree.nodes_accessed_kv
    # nodes_accessed_ssm = radix_tree.nodes_accessed_ssm
    # print(f"nodes_created {nodes_created}, len(token_representations) {len(token_representations)}, len(nodes_accessed_kv) {len(nodes_accessed_kv)}, len(nodes_accessed_ssm) {len(nodes_accessed_ssm)}")


    # run policies that don't require a config sweep: SGLang+ (V1) and Marconi (V2)
    for policy_v in [1, 2]:
        configs = {
            "trace_name": trace_name,
            "num_sessions": num_sessions,
            "capacity_bytes": capacity_bytes,
            "evict_policy_version": policy_v,
            "eff_weight": 0.0,
            "sessions_per_second": sessions_per_second,
            "avg_response_time": avg_response_time,
            "num_ssm_layers": num_ssm_layers,
            "num_attn_layers": num_attn_layers,
            "num_mlp_layers": num_mlp_layers,
            "d": d,
            "n": n,
            "bootstrap_multiplier": args.bootstrap_multiplier,
        }
        output, local_args, radix_tree, request_hit_rate, token_hit_rate, total_mamba_flop_savings, total_attn_flop_savings, total_mlp_flop_savings = \
            run_trace_with_config(**configs)
        total_flops_savings = total_mamba_flop_savings + total_attn_flop_savings + total_mlp_flop_savings
        locals()[f"v{policy_v}_max_hit_rate"], locals()[f"v{policy_v}_max_flops_saved"] = token_hit_rate, total_flops_savings
        locals()[f"v{policy_v}_request_history"] = radix_tree.request_history
        print(f"V{policy_v}: hit rate {token_hit_rate*100:.2f}%, FLOPs saved {total_flops_savings/1e12:.2f}e12")

    # V3: offline-optimal, static-alpha oracle
    v3_hit_rate_list, v3_flops_saved_list = [], []
    weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        
    configs = [{
        "trace_name": trace_name,
        "num_sessions": num_sessions,
        "capacity_bytes": capacity_bytes,
        "evict_policy_version": 3,
        "eff_weight": eff_weight,
        "sessions_per_second": sessions_per_second,
        "avg_response_time": avg_response_time,
        "num_ssm_layers": num_ssm_layers,
        "num_attn_layers": num_attn_layers,
        "num_mlp_layers": num_mlp_layers,
        "d": d,
        "n": n,
        "bootstrap_multiplier": args.bootstrap_multiplier,
    }
    for eff_weight in weights]
    
    
    with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
        futures = [executor.submit(run_trace_with_config, **config) for config in configs]
        
        # for future in as_completed(futures):
        for future in futures:
            try:
                # output = future.result()[0]
                output, local_args, radix_tree, request_hit_rate, token_hit_rate, total_mamba_flop_savings, total_attn_flop_savings, total_mlp_flop_savings = future.result()
                total_flops_savings = total_mamba_flop_savings + total_attn_flop_savings + total_mlp_flop_savings
                # print(output)  # Print the output of each process after it finishes
                
                # add to results
                v3_hit_rate_list.append(token_hit_rate)
                v3_flops_saved_list.append(total_flops_savings)
                
                print(f"V3: eff_weight {local_args['eff_weight']}, hit rate {token_hit_rate*100:.2f}%, FLOPs saved {total_flops_savings/1e12:.2f}e12")
                
            except Exception as e:
                print(f"Evaluation raised an exception: {e}")
            
            # Explicitly delete the result to release memory
            del future, output, local_args, radix_tree, request_hit_rate, token_hit_rate, total_mamba_flop_savings, total_attn_flop_savings, total_mlp_flop_savings
            gc.collect()  # Force garbage collection

    v3_max_hit_rate = max(v3_hit_rate_list)
    v3_max_flops_saved = max(v3_flops_saved_list)
    print(f"V3: hit rate {v3_max_hit_rate*100:.2f}%, FLOPs saved {v3_max_flops_saved/1e12:.2f}e12")
    
    for i in [2, 3]:
        hit_rate_win = (locals()[f"v{i}_max_hit_rate"] - locals()["v1_max_hit_rate"]) / locals()["v1_max_hit_rate"]  # absolute win
        flops_saved_win = (locals()[f"v{i}_max_flops_saved"] - locals()["v1_max_flops_saved"]) / locals()["v1_max_flops_saved"]
        print(f"V{i} compared to V1: hit_rate_win {hit_rate_win*100:.2f}%, flops_saved_win {flops_saved_win*100:.2f}%")
    
        if "vllm_max_hit_rate" in locals():
            hit_rate_win = (locals()[f"v{i}_max_hit_rate"] - locals()["vllm_max_hit_rate"]) / locals()["vllm_max_hit_rate"]  # absolute win
            flops_saved_win = (locals()[f"v{i}_max_flops_saved"] - locals()["vllm_max_flops_saved"]) / locals()["vllm_max_flops_saved"]
            print(f"V{i} compared to vLLM: hit_rate_win {hit_rate_win*100:.2f}%, flops_saved_win {flops_saved_win*100:.2f}%")
    if "vllm_max_hit_rate" in locals():
        hit_rate_win = (locals()[f"v1_max_hit_rate"] - locals()["vllm_max_hit_rate"]) / locals()["vllm_max_hit_rate"]  # absolute win
        flops_saved_win = (locals()[f"v1_max_flops_saved"] - locals()["vllm_max_flops_saved"]) / locals()["vllm_max_flops_saved"]
        print(f"V1 compared to vLLM: hit_rate_win {hit_rate_win*100:.2f}%, flops_saved_win {flops_saved_win*100:.2f}%")
    
    # persist results to pickle file
    experiment_data = {}
    policies = ["vllm", "v1", "v2"]
    metrics = ["max_hit_rate", "max_flops_saved", "request_history"]
    for p, m in itertools.product(policies, metrics):
        if f"{p}_{m}" in locals():
            experiment_data[f"{p}_{m}"] = locals()[f"{p}_{m}"]
    experiment_data["time"] = datetime.now(timezone('EST')).strftime('%Y-%m-%d %H:%M:%S %Z')
    trace_filename = f"{trace_name}_sps={sessions_per_second}"
    if trace_name in ["swebench"]:
        trace_filename += f"_art={avg_response_time}"
    trace_filename += f"_nums={num_sessions}"
    if not os.path.exists(f"./results/{capacity_bytes}"):
        os.makedirs(f"./results/{capacity_bytes}")
    if not os.path.exists(f"./results/{capacity_bytes}/{trace_filename}"):
        os.makedirs(f"./results/{capacity_bytes}/{trace_filename}")
    with open(f"./results/{capacity_bytes}/{trace_filename}/{DATETIME}.pickle", "wb") as f:
        pickle.dump(experiment_data, f)

    print(f"Results stored in ./results/{capacity_bytes}/{trace_filename}/{DATETIME}.pickle")

# %%
