# %%
import gc
import os
import copy
import time
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Automatically get number of CPUs available for parallelizing grid search
NUM_CPUS = int(os.cpu_count())

def replay_trace(
    radix_tree,
    past_requests,
    evict_policy_version: int,
    eff_weight: float,
):
    """Replays a request arrival trace of a radix tree, given the 
    radix tree configs.

    Args:
        radix_tree (RadixCache): radix tree for prefix cache
        past_requests (list): list of past requests
        evict_policy_version (int): eviction policy version number
        eff_weight (float): alpha

    Returns:
        RadixCache: radix tree that experienced all past requests
    """
    start_time = time.time()
    radix_tree.evict_policy_version = evict_policy_version
    radix_tree.eff_weight = eff_weight
    
    num_requests = len(past_requests)
    for request_id in range(num_requests):
        input_token_ids, output_token_ids = past_requests[request_id]
        # retrieve prefix
        radix_tree.match_prefix(input_token_ids)
        # insert
        radix_tree.insert(
            token_ids=input_token_ids + output_token_ids,
            state_at_leaf=request_id,
            state_at_branchoff=request_id,
        )
    end_time = time.time()
    # print(f"Replay trace took {end_time - start_time:.1f}s")
    return radix_tree


class ConfigTuner:
    def __init__(
        self,
    ):
        self.tree_snapshot = None  # snapshot of radix tree at the beginning of the past window
        self.num_tunings = 0
    
    def tune_config(
        self,
        request_history_windowed: list,
        weights: list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    ):
        """Runs grid search to find the best weight in the past window,
        and then update the weight in the tree.

        Args:
            request_history_windowed (list): List of requests in the past window.
            weights (list): List of possible alphas to do a grid search upon.

        Returns:
            best_eff_weight: The best alpha in the grid search.
        """
        start_time = time.time()
        print(f"Running config tuning")
        
        results = {}  # weight: (hit_rate, flops_saved)
        configs = [{
            "radix_tree": pickle.loads(pickle.dumps(self.tree_snapshot, -1)),  # faster copy.deepcopy(self.tree_snapshot)
            "past_requests": request_history_windowed,
            "evict_policy_version": 3,
            "eff_weight": eff_weight,
        } for eff_weight in weights]
        
        with ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
            before_submitting_time = time.time()
            futures = [executor.submit(replay_trace, **config) for config in configs]
            after_submitting_time = time.time()
            
            for future in as_completed(futures):
                try:
                    radix_tree = future.result()
                    
                    # extract the hit rate of the request trace given a specific tree config
                    request_hit_rate, token_hit_rate, \
                        total_mamba_flop_savings, total_attn_flop_savings, total_mlp_flop_savings = \
                            radix_tree.get_cache_stats(verbose=False, last_n=radix_tree.tuning_interval)  # NOTE(ruipan): removing last_n has a negligible impact on Marconi's results
                    total_flops_savings = total_mamba_flop_savings + total_attn_flop_savings + total_mlp_flop_savings
                    eff_weight = radix_tree.eff_weight
                    
                    results[eff_weight] = (request_hit_rate, token_hit_rate, total_flops_savings,)
                    print(f"Tuning: eff_weight {eff_weight}, hit rate {token_hit_rate*100:.2f}%, FLOPs saved {total_flops_savings/1e12:.2f}e12")
                except Exception as e:
                    print(f"Evaluation raised an exception:")
                    traceback.print_exc()  # this prints the full traceback
                
                # del radix_tree
                # gc.collect()
        
        # find the best performing eff_weight
        best_eff_weight = max(results, key=lambda k: results[k][2])
        # print(f"Round {self.num_tunings}, best_eff_weight {best_eff_weight}")
        self.num_tunings += 1
        
        end_time = time.time()
        timestamps = [x - start_time for x in [start_time, before_submitting_time, after_submitting_time]]
        print(f"Tuning config took {end_time - start_time:.2f}s")
        
        return best_eff_weight

# %%
