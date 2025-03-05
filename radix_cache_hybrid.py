# %%
import copy
import heapq
import time
import itertools
import statistics
from datetime import datetime
from collections import defaultdict
from scipy.stats import rankdata
from transformers import AutoTokenizer, AutoModelForCausalLM

from config_tuner import ConfigTuner
from utils import (
    get_attn_flops, get_flops_efficiency, get_kvs_size, get_mamba1_flops,
    get_mamba_state_size, get_mlp_flops, get_model_state_size,
    _key_match, _normalize,
)


class HybridStates:
    def __init__(
        self,
        input_ids: list = None,
        mamba_states=None,  # includes both ssm and conv states
        kv_cache=None,  # includes tokens from parent node to current node
    ):
        self.input_ids = input_ids
        self.mamba_states = mamba_states
        self.kv_cache = kv_cache
    

class TreeNode:
    def __init__(self, logical_ts=None):
        self.key = None  # tuple of token IDs between the node's parent and itself
        self.value = None  # list of token IDs between the node's parent and itself
        self.all_token_ids = None  # list of token IDs represented by the current node

        self.parent = None
        self.children = defaultdict(TreeNode)  # key: ID of the first token of the child's key. value: child node.

        self.last_access_time = time.time() if logical_ts is None else logical_ts
        self.hybrid_states = None
    
    def get_all_token_ids(self):
        node = self
        token_ids_of_nodes = [self.value]  # token IDs of child, parent, grandparent, ...
        while node.parent is not None and node.parent.value is not None:  # track up and get the bottom-up order of nodes
            node = node.parent
            token_ids_of_nodes.append(node.value)
        token_ids_of_nodes = list(reversed(token_ids_of_nodes))
        all_token_ids = list(itertools.chain.from_iterable(token_ids_of_nodes))  # join list of lists
        # print(f"token_ids_of_nodes {token_ids_of_nodes}")
        return all_token_ids
        
    def __str__(self):
        token_ids = []
        node = self
        while node.parent is not None:  # track up and get the bottom-up order of nodes
            token_ids.append(list(node.key))  # tuple -> list
            node = node.parent
        token_ids = reversed(token_ids)  # bottom-up -> top-down
        token_ids = [j for i in token_ids for j in i]  # concatenate list of lists
        return f"TreeNode: tokens {str(token_ids)}, last accessed {datetime.fromtimestamp(self.last_access_time).strftime('%I:%M:%S.%f')}"
    
    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


class RadixCache:
    def __init__(
        self, 
        # model specs: by default, uses NVIDIA's 7B Hybrid model
        num_ssm_layers: int = 24,
        num_attn_layers: int = 4,
        num_mlp_layers: int = 28,
        d: int = 4096,  # D
        n: int = 128,  # N
        capacity_bytes=1e9,  # in bytes
        
        use_logical_ts=True,  # use logical timestamps for exact reproducibility
        
        evict_policy_version=2,
        eff_weight=0,
        bootstrap_multiplier=5,
        
        *args, **kwargs,
    ):
        self.root_node = TreeNode()
        self.num_nodes = 0  # not counting the root node.  # NOTE(ruipan): buggy?
        
        self.num_ssm_layers = num_ssm_layers
        self.num_attn_layers = num_attn_layers
        self.num_mlp_layers = num_mlp_layers
        self.d = d
        self.n = n
        
        self.capacity_bytes = capacity_bytes  # total cache size available in bytes. Defaults to 1GB.
        
        # for every request, tuple of: (cache hit or not, total tokens in request, tokens reused/skipped prefill)
        self.request_history = []
        
        self.use_logical_ts = use_logical_ts
        if use_logical_ts:
            self.logical_ts = 0
            self.time_increment = 1
        
        # NOTE(ruipan): When adding a new eviction policy, search for "self.evict_policy_version" and make changes where necessary
        self.evict_policy_version = evict_policy_version
        self.eff_weight = eff_weight
        
        # specific to Marconi's retrospective config tuning of alpha
        self.config_tuner = ConfigTuner()
        self.tuning_interval = 500  # NOTE(ruipan): legacy code for continuous tuning. Used in config tuner.
        self.request_history_windowed = []  # stores token-level request info in the past window. list of [input_tokens, output_tokens]
        
        self.num_reqs_before_eviction = None  # number of requests the cache could house prior to the first eviction
        self.bootstrap_multiplier = bootstrap_multiplier
        self.bootstrap_window_size = None  # bootstrap_multiplier * num_reqs_before_eviction
    
    def insert(
        self,
        token_ids,  # list of token IDs as integers
        state_at_leaf,  # state after the last decoded token
        state_at_branchoff=None,  # state at the branchoff node
    ):
        """Insert a sequence of tokens into the radix tree.

        Args:
            token_ids (list): list of token IDs as integers
            state_at_leaf (HybridStates): state after the last decoded token
            state_at_branchoff (HybridStates, optional): state at the branchoff node. 
                Required if insertion will create a branchoff node. Defaults to None.
        """
        if self.use_logical_ts:
            self.logical_ts += self.time_increment
        
        if self.evict_policy_version in [2]:
            # add to recent request history
            assert self.request_history_windowed[-1][1] is None
            input_token_ids = self.request_history_windowed[-1][0]
            input_len = _key_match(input_token_ids, token_ids)
            assert len(input_token_ids) == input_len  # token_ids = input_token_ids + output_token_ids
            output_token_ids = token_ids[input_len:]
            self.request_history_windowed[-1][1] = output_token_ids
            
            # check if tuning is necessary
            if (self.evict_policy_version == 2 and self.bootstrap_window_size is not None and len(self.request_history) == self.bootstrap_window_size):
                # do tuning
                print(f"len(self.request_history_windowed) {len(self.request_history_windowed)}")
                best_eff_weight = self.config_tuner.tune_config(self.request_history_windowed)
                self.eff_weight = best_eff_weight
                # reset windowed history
                self.config_tuner.tree_snapshot = copy.deepcopy(self)
                self.request_history_windowed = []
                print(f"V2: Updated eff_weight to be {best_eff_weight}")
        
        # check for enough space. if not, evict
        _, _, branchoff_required, prefix_len = self.match_prefix(token_ids, actually_inserting=False)
        if branchoff_required:
            assert state_at_branchoff is not None, f"Insertion will create a branchoff node, but a state is not provided"
    
        num_extra_tokens = len(token_ids) - prefix_len  # needs to store their KVs
        num_extra_mamba_states = 2 if branchoff_required else 1
        bytes_needed = num_extra_mamba_states * self.num_ssm_layers * get_mamba_state_size(self.d, self.n) + \
            self.num_attn_layers * get_kvs_size(num_extra_tokens, self.d)  # to insert this new sequence
        
        if self.get_tree_size() + bytes_needed > self.capacity_bytes:  # insertion will lead to overflow
            bytes_to_remove = self.get_tree_size() + bytes_needed - self.capacity_bytes
            self.evict(bytes_to_remove=bytes_to_remove)
        
        self._insert_helper(
            node=self.root_node, 
            key=tuple(token_ids),  # list -> tuple to make it hashable
            value=token_ids,
            state_at_leaf=state_at_leaf,
            state_at_branchoff=state_at_branchoff,
        )
    
    def pretty_print(self, verbose=False):
        self._print_helper(self.root_node, 0, verbose=verbose)
    
    def get_tree_size(self, verbose=False):
        mamba_state_size, attn_state_size = self.get_state_size()
        total_size = mamba_state_size + attn_state_size
        if verbose: 
            print(f"Tree total size: {total_size/1e9:.2f} GB (Mamba {mamba_state_size/1e9:.2f} GB, Attn {attn_state_size/1e9:.2f} GB)", flush=True)
        return total_size
        
    def get_state_size(self):
        num_cached_mamba_states, num_cached_kv_tokens = self.get_num_cached_tokens()
        mamba_state_size = self.num_ssm_layers * num_cached_mamba_states * get_mamba_state_size(self.d, self.n)
        attn_state_size = self.num_attn_layers * get_kvs_size(num_cached_kv_tokens, self.d)
        return mamba_state_size, attn_state_size
        
    def get_cache_stats(self, verbose=True, last_n=None):
        if last_n is not None:  # only getting the stats of the n most recent requests
            request_history = self.request_history[-last_n:]
        else:  # stats of the whole lifespan of the cache
            request_history = self.request_history
        num_requests_recorded = len(request_history)
        num_total_tokens = sum([x[1] for x in request_history])
        num_tokens_saved = sum([x[2] for x in request_history])
        
        total_mamba_flop_savings = self.num_ssm_layers * get_mamba1_flops(l=num_tokens_saved, d=self.d, n=self.n)
        total_attn_flop_savings = sum([self.num_attn_layers * get_attn_flops(l=x[2], d=self.d) for x in request_history])
        total_mlp_flop_savings = sum([self.num_mlp_layers * get_mlp_flops(l=x[2], d=self.d) for x in request_history])
        
        request_hit_rate = sum([x[0] for x in request_history]) / num_requests_recorded
        token_hit_rate = num_tokens_saved / num_total_tokens
        
        if verbose:
            print(f"request_hit_rate {request_hit_rate*100:.2f}%, token_hit_rate {token_hit_rate*100:.2f}%", flush=True)
            print(f"Total FLOPs saved: {(total_mamba_flop_savings + total_attn_flop_savings + total_mlp_flop_savings)/1e12:.2f}e12", flush=True)
        return request_hit_rate, token_hit_rate, total_mamba_flop_savings, total_attn_flop_savings, total_mlp_flop_savings
    
    def get_num_cached_tokens(self):
        # num_cached_mamba_states, num_cached_kv_tokens
        return self._get_num_cached_tokens_helper(self.root_node)
    
    def match_prefix(self, input_token_ids, actually_inserting=True):
        """Given a new request, check how much prefix can be reused.
        Also checks if we can identify a soon-to-be branchoff node, so that we can checkpoint it during prefill.
        
        Args:
            input_token_ids (list): list of token IDs as integers
            actually_inserting (bool, optional): whether match_prefix() is used for a cache lookup. Defaults to True.

        Returns:
            prefix_token_ids: actually reusable tokens. Can only reuse if a prefix of the input tokens can be represented by a series of existing radix nodes.
            nodes_accessed: list of pointers to the nodes with the prefixes
            branchoff_required: whether inserting the current sequence will result in a branchoff node
            prefix_len: actual prefix length, ignoring reusability
        """
        if self.use_logical_ts:
            self.logical_ts += self.time_increment
        
        if self.evict_policy_version in [2] and actually_inserting and len(self.request_history) == 0:
            self.config_tuner.tree_snapshot = copy.deepcopy(self)

        prefix_token_ids = []
        nodes_accessed = [self.root_node]
        prefix_len = self._match_prefix_helper(self.root_node, input_token_ids, prefix_token_ids, nodes_accessed)
        # reusable tokens represented by cached states
        prefix_token_ids = [j for i in prefix_token_ids for j in i]  # concatenate list of lists
        num_tokens_skipped = len(prefix_token_ids)

        branchoff_required = prefix_len > len(prefix_token_ids)  # actual prefix is longer -> insertion will create branchoff node
        
        cache_hit = len(prefix_token_ids) > 0
        
        if actually_inserting:
            self.request_history.append((cache_hit, len(input_token_ids), num_tokens_skipped,))
            if self.evict_policy_version in [2]:
                self.request_history_windowed.append([input_token_ids, None])  

        if self.evict_policy_version in [2, 3]:  # Only update the timestamp of a single node on prefix matching
            if len(nodes_accessed) != 1:  # nodes other than the root node was used for prefix reusing
                if not self.use_logical_ts:
                    nodes_accessed[-1].last_access_time = time.time()
                else:
                    nodes_accessed[-1].last_access_time = self.logical_ts
        
        return prefix_token_ids, nodes_accessed, branchoff_required, prefix_len

    def get_flops_efficiency_list(self):
        return [get_flops_efficiency(l, self.d, self.n, self.num_ssm_layers, self.num_attn_layers, self.num_mlp_layers) for _, l, _ in self.request_history]

    def _insert_helper(self, node, key, value, state_at_leaf, state_at_branchoff):   
        if self.evict_policy_version in [1]:  # SGLang-like, all ancestor nodes' timestamps are updated on insertion/prefix matching
            if not self.use_logical_ts:
                node.last_access_time = time.time()
            else:
                node.last_access_time = self.logical_ts
        elif self.evict_policy_version in [2, 3]:  # Only update the timestamp of a single node on prefix matching
            pass
        
        if len(key) == 0:
            return 0
        
        if key[0] in node.children.keys():
            # at least one token is successfully matched
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)

            if prefix_len == len(child.key):
                if prefix_len == len(key):
                    # perfect matching: input is already recorded in the tree
                    return prefix_len
                else:
                    # input is longer than key of child
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value, state_at_leaf, state_at_branchoff)

            # need to branch off from child: split child and insert the diff
            assert state_at_branchoff is not None, f"Splitting node but the state at the branchoff is None"
            new_node = self._split_node(child.key, child, prefix_len, state_at_branchoff)
            return prefix_len + self._insert_helper(
                new_node, key[prefix_len:], value[prefix_len:],
                state_at_leaf, None,
                # state at branchoff is None because each insertion will only trigger
                # one node splitting
            )

        if len(key):  # no further tokens can be matched, insert leaf node
            # bytes_needed = self.num_ssm_layers * get_mamba_state_size(self.d, self.n) + \
            #     self.num_attn_layers * get_kvs_size(len(value), self.d)
            # if self.get_tree_size() + bytes_needed > self.capacity_bytes:
            #     self.evict(bytes_to_remove=self.get_tree_size()+bytes_needed-self.capacity_bytes)
            # print(f"creating new node in _insert_helper")
            logical_ts = self.logical_ts if self.use_logical_ts else None
            new_node = TreeNode(logical_ts=logical_ts)
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            new_node.hybrid_states = state_at_leaf
            node.children[key[0]] = new_node
            # print(f"_insert_helper: self.num_nodes += 1, previously {self.num_nodes}")
            self.num_nodes += 1
        return 0
    
    def _split_node(self, key, child: TreeNode, split_len, state_at_branchoff):
        # bytes_needed = self.num_ssm_layers * get_mamba_state_size(self.d, self.n)
        # if self.get_tree_size() + bytes_needed > self.capacity_bytes:
        #     self.evict(bytes_to_remove=self.get_tree_size()+bytes_needed-self.capacity_bytes)
        # print(f"creating new node in _split_node")
        
        # new_node -> child
        logical_ts = self.logical_ts if self.use_logical_ts else None
        new_node = TreeNode(logical_ts=logical_ts)
        new_node.children = {key[split_len:][0]: child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        new_node.hybrid_states = state_at_branchoff
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[key[:split_len][0]] = new_node
        # print(f"_split_node(): new branchoff node created, Mamba states required for: {new_node}")
        # print(f"_split_node: self.num_nodes += 1, previously {self.num_nodes}")
        self.num_nodes += 1
        return new_node
    
    def _match_prefix_helper(self, node, key, value, nodes_accessed): 
        if self.evict_policy_version in [1]:  # SGLang-like, all ancestor nodes' timestamps are updated on insertion/prefix matching
            if not self.use_logical_ts:
                node.last_access_time = time.time()
            else:
                node.last_access_time = self.logical_ts
        elif self.evict_policy_version in [2, 3]:  # Only update the timestamp of a single node on prefix matching
            pass
        
        if len(key) == 0 or node is None:
            return 0

        if key[0] in node.children.keys():  # one of the child shares some partial prefix
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):  # inserting the current sequence will lead to branching off
                # print(f"prefix_len {prefix_len}, len(child.key) {len(child.key)}")
                return prefix_len + self._match_prefix_helper(child, key[prefix_len:], value, nodes_accessed)
            else:  # child's child potentially has more prefixes to match
                value.append(child.value)
                nodes_accessed.append(child)
                return prefix_len + self._match_prefix_helper(child, key[prefix_len:], value, nodes_accessed)
        
        return 0

    def _collect_leaves(self):
        ret_list = []

        def dfs_(cur_node):
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list

    def _collect_leaf_and_single_child_nodes(self):
        ret_list = []
        
        def dfs_(cur_node):
            if len(cur_node.children) in [0, 1] and cur_node.value is not None:  # exclude the root node
                ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list

    def _collect_keystone_nodes(self):
        ret_list = []

        def dfs_(cur_node):
            if len(cur_node.children) >= 1:
                ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list
    
    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        # print(f"_delete_leaf: self.num_nodes -= 1, previously {self.num_nodes}")
        self.num_nodes -= 1
    
    def _evict_intermediate_node(self, node):
        # absorbs its child
        assert len(node.children) == 1
        
        child = node.children[list(node.children.keys())[0]]
        
        new_node = TreeNode()
        new_node.last_access_time = max(node.last_access_time, child.last_access_time)
        new_node.children = child.children
        new_node.parent = node.parent
        new_node.value = node.value + child.value
        new_node.key = tuple(new_node.value)
        new_node.hybrid_states = child.hybrid_states  # NOTE(ruipan): needs more manipulations on the kvs
        node.parent.children[node.key[0]] = new_node
        
        del node
        # print(f"_evict_intermediate_node: self.num_nodes -= 1, previously {self.num_nodes}")
        self.num_nodes -= 1
            
    def _print_helper(self, node: TreeNode, depth=0, verbose=False):
        if depth == 0:
            print(f"num_nodes in tree: {self.num_nodes}")
        separator = "  "  # "\t"
        for _, child in node.children.items():
            if type(child.hybrid_states) is int:  # for debugging
                hybrid_states_str = str(child.hybrid_states)
            elif type(child.hybrid_states) is HybridStates:
                hybrid_states_str = f"{len(child.hybrid_states.input_ids)} tokens"
            else:
                assert False
            if not verbose:
                print(separator * depth, "├─", len(child.key), f"last accessed {datetime.fromtimestamp(child.last_access_time).strftime('%I:%M:%S.%f')}, state: {hybrid_states_str}")
            else:  # print the token IDs associated with the node
                print(separator * depth, "├─", len(child.key), f"child.key {child.key}, last accessed {datetime.fromtimestamp(child.last_access_time).strftime('%I:%M:%S.%f')}, state: {hybrid_states_str}")
            self._print_helper(child, depth=depth + 1, verbose=verbose)
        
    def _get_num_cached_tokens_helper(self, node: TreeNode):
        if node.value is not None:  
            total_num_cached_mamba_states = 1
            total_num_cached_kv_tokens = len(node.value)
        else:  # root node
            total_num_cached_mamba_states, total_num_cached_kv_tokens = 0, 0
        for _, child in node.children.items():
            num_cached_mamba_states, num_cached_kv_tokens = self._get_num_cached_tokens_helper(child)
            total_num_cached_mamba_states += num_cached_mamba_states
            total_num_cached_kv_tokens += num_cached_kv_tokens
        return total_num_cached_mamba_states, total_num_cached_kv_tokens
    
    def evict(self, bytes_to_remove):
        if self.use_logical_ts:
            self.logical_ts += self.time_increment
        
        if self.num_reqs_before_eviction is None:
            self.num_reqs_before_eviction = len(self.request_history)
            # self.bootstrap_window_size = 5 * self.num_reqs_before_eviction
            self.bootstrap_window_size = self.bootstrap_multiplier * self.num_reqs_before_eviction
            
            if self.evict_policy_version in [2]:
                print(f"V2: Evict() first called, has seen {len(self.request_history)} requests, setting bootstrap_window_size to be {self.bootstrap_window_size}")
            
        if self.evict_policy_version in [1]:  # SGLang+: LRU on the leaf nodes
            self.evict_v1(bytes_to_remove=bytes_to_remove)
        elif self.evict_policy_version in [2, 3]:  # all nodes with <=1 children are considered for eviction
            self.evict_v2(bytes_to_remove=bytes_to_remove)
        else:
            raise NotImplementedError(f"self.evict_policy_version is {self.evict_policy_version}")
    
    def evict_v1(self, bytes_to_remove):
        # only evict leaf nodes
        leaves = self._collect_leaves()
        heapq.heapify(leaves)  # leaves is now a min-heap that pops the least recently accessed leaf
        
        bytes_evicted = 0
        while bytes_evicted < bytes_to_remove and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break

            bytes_evicted += self.num_ssm_layers * get_mamba_state_size(self.d, self.n) + self.num_attn_layers * get_kvs_size(len(x.value), self.d)
            # print(f"Eviction: Evicting node with keylen {len(x.key)}")
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def evict_v2(self, bytes_to_remove):
        bytes_evicted = 0
        
        while bytes_evicted < bytes_to_remove and self.num_nodes > 0:
            # collect all nodes with <=1 child
            leaf_and_single_child_nodes = self._collect_leaf_and_single_child_nodes()

            if not self.use_logical_ts:
                current_ts = time.time()
            else:
                current_ts = self.logical_ts
            
            timestamps = [node.last_access_time for node in leaf_and_single_child_nodes]
            flops_efficiency = []
            for node in leaf_and_single_child_nodes:
                seqlen_total = len(node.get_all_token_ids())
                seqlen_child = len(node.value)
                seqlen_parent = seqlen_total - seqlen_child
                
                if self.evict_policy_version in [2, 3]:
                    # savings are relative to the parent
                    flops_savings_mamba = self.num_ssm_layers * get_mamba1_flops(seqlen_child, self.d, self.n)
                    flops_savings_attn = self.num_attn_layers * (get_attn_flops(seqlen_total, self.d) - get_attn_flops(seqlen_parent, self.d))
                    flops_savings_mlp = self.num_mlp_layers * (get_mlp_flops(seqlen_total, self.d) - get_attn_flops(seqlen_parent, self.d))
                    total_flops_savings = flops_savings_mamba + flops_savings_attn + flops_savings_mlp
                    total_memory = self.num_ssm_layers * get_mamba_state_size(self.d, self.n) + self.num_attn_layers * get_kvs_size(seqlen_total, self.d)
                else:
                    raise NotImplementedError(f"Using evict_v2 but policy version is {self.evict_policy_version}!")
                flops_efficiency.append(total_flops_savings / total_memory)

            # scores: higher is gbetter
            normalized_flops_efficiency = _normalize(flops_efficiency)
            normalized_recency_scores = _normalize([1 / (current_ts - ts) for ts in timestamps])

            if self.evict_policy_version in [2, 3]:
                eff_weight = self.eff_weight
            else:
                raise NotImplementedError(f"Using evict_v2 but policy version is {self.evict_policy_version}!")
            
            if self.evict_policy_version in [2, 3]:
                utility_scores = [  # higher is hotter and more flops efficient
                    eff_weight * eff_score + recency_score
                    for (eff_score, recency_score) in zip(normalized_flops_efficiency, normalized_recency_scores)
                ]
            else:
                raise NotImplementedError(f"Using evict_v2 but policy version is {self.evict_policy_version}!")

            
            node_to_evict = leaf_and_single_child_nodes[utility_scores.index(min(utility_scores))]
            
            assert len(node_to_evict.children) in [0, 1]
            if len(node_to_evict.children) == 0:
                bytes_evicted += self.num_ssm_layers * get_mamba_state_size(self.d, self.n) + self.num_attn_layers * get_kvs_size(len(node.value), self.d)
                self._delete_leaf(node_to_evict)
            else:
                bytes_evicted += self.num_ssm_layers * get_mamba_state_size(self.d, self.n)
                self._evict_intermediate_node(node_to_evict)

