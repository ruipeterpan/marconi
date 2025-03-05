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
from utils import _key_match, get_attn_flops, get_mlp_flops, get_mamba1_flops, get_kvs_size, get_mamba_state_size, get_model_state_size


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

        self.parent = None
        self.children = {}  # key: tuple of the child's key. value: child node.

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

    def get_depth(self):
        if self.parent == None:  # root node
            return 0
        else:
            return 1 + self.parent.get_depth()
        
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
        # model specs
        num_ssm_layers: int = 24,
        num_attn_layers: int = 4,
        num_mlp_layers: int = 28,
        d: int = 4096,  # D
        n: int = 128,  # N
        capacity_bytes=1e9,  # in bytes
        use_logical_ts=True,  # use logical timestamps for reproducibility
        block_size=32,
        *args, **kwargs,
    ):
        """This prefix tree builds atop a radix tree but effectively implements vLLM's default, token-block-based eviction policy:
        - Only evict the token blocks with ref_count == 0
        - Prefers to first evict older blocks following LRU
        - If the timestamp is the same, free the block with longer prefix lengths first (starting from leaf nodes)
        
        Reference: https://github.com/vllm-project/vllm/issues/2614

        Args:
            num_ssm_layers (int, optional): Number of SSM layers in the model. Defaults to 24.
            num_attn_layers (int, optional): Number of Attention layers in the model. Defaults to 4.
            num_mlp_layers (int, optional): Number of MLP layers in the model. Defaults to 28.
            d (int, optional): Model dimension. Defaults to 4096.
        """
        self.root_node = TreeNode()
        
        self.num_ssm_layers = num_ssm_layers
        self.num_attn_layers = num_attn_layers
        self.num_mlp_layers = num_mlp_layers
        self.d = d
        self.n = n
        
        self.capacity_bytes = capacity_bytes  # total cache size available in bytes. Defaults to 1GB.
        
        # for every request, tuple of: (cache hit or not, total tokens in request, tokens saved)
        self.request_history = []
        
        self.use_logical_ts = use_logical_ts
        if use_logical_ts:
            self.logical_ts = 0
            self.time_increment = 1
        
        # vLLM+
        self.block_size = block_size  # store a state every 8, 16, 32 tokens
        
        # # for motivational experiment
        # self.nodes_created = 0
        # self.token_representations = []
        # # self.node_accessed_kv = 0
        # # self.nodes_accessed_ssm = 0
        # self.nodes_accessed_kv = []
        # self.nodes_accessed_ssm = []
    
    def insert(
        self,
        token_ids,  # list of token IDs as integers
        *args, **kwargs,
    ):
        """Insert a sequence of tokens into the radix tree.

        Args:
            token_ids (list): list of token IDs as integers
        """
        if self.use_logical_ts:
            self.logical_ts += self.time_increment

        # check for enough space. if not, evict
        prefix_token_ids, nodes_accessed, _ = self.match_prefix(token_ids, actually_inserting=False)
        # estimate the space needed. maybe off by one or two token blocks, but it's fine.
        num_extra_tokens = len(token_ids) - self.block_size * len(nodes_accessed)  # needs to store their KVs
        num_extra_mamba_states = num_extra_tokens / self.block_size
        bytes_needed = num_extra_mamba_states * self.num_ssm_layers * get_mamba_state_size(self.d, self.n) + \
            self.num_attn_layers * get_kvs_size(num_extra_tokens, self.d)  # to insert this new sequence
        
        if self.get_tree_size() + bytes_needed > self.capacity_bytes:  # insertion will lead to overflow
            bytes_to_remove = self.get_tree_size() + bytes_needed - self.capacity_bytes
            self.evict(bytes_to_remove=bytes_to_remove)
        
        self._insert_helper(
            node=self.root_node, 
            key=tuple(token_ids),  # list -> tuple to make it hashable
            value=token_ids,
        )
    
    def pretty_print(self):
        self._print_helper(self.root_node, 0)
    
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

        prefix_token_ids = []
        nodes_accessed = [self.root_node]
        prefix_len = self._match_prefix_helper(self.root_node, input_token_ids, prefix_token_ids, nodes_accessed)
        # reusable tokens represented by cached states
        prefix_token_ids = [j for i in prefix_token_ids for j in i]  # concatenate list of lists
        num_tokens_skipped = len(prefix_token_ids)
        cache_hit = len(prefix_token_ids) > 0
        
        if actually_inserting:
            self.request_history.append((cache_hit, len(input_token_ids), num_tokens_skipped,))
        
        # # for motivational experiment
        # if len(nodes_accessed) > 1:  # accessed some nodes with prefixes
        #     # self.node_accessed_kv += (len(nodes_accessed) - 1)  # doesn't account for duplicated accesses
        #     # self.nodes_accessed_ssm += 1
        #     # print(f"len(nodes_accessed) {len(nodes_accessed)}")
        #     for i, node in enumerate(nodes_accessed):
        #         if i == 0:  continue  # don't need to account for the root node
        #         token_representation_hash = hash(tuple(node.get_all_token_ids()))
        #         if token_representation_hash not in self.nodes_accessed_kv:
        #             self.nodes_accessed_kv.append(token_representation_hash)
        #         if i == len(nodes_accessed) - 1:  # only the SSM states of the last node accessed is really needed
        #             if token_representation_hash not in self.nodes_accessed_ssm:
        #                 self.nodes_accessed_ssm.append(token_representation_hash)
        
        return prefix_token_ids, nodes_accessed, prefix_len


    def _insert_helper(self, node, key, value):   
        if not self.use_logical_ts:
            node.last_access_time = time.time()
        else:
            node.last_access_time = self.logical_ts

        # locate the node to insert into
        max_prefix_len = 0
        if len(node.children) != 0:  # node has children. first check if the target node is one of the children.
            for c_key, child in node.children.items():  # c_key: list of token IDs. child: reference to node.
                if _key_match(c_key, key) > max_prefix_len:
                    max_prefix_len = _key_match(c_key, key)
            for c_key, child in node.children.items():  # c_key: list of token IDs. child: reference to node.
                prefix_len = _key_match(c_key, key)
                if prefix_len == len(c_key) == max_prefix_len:  # child is a partial block and is the prefix of the sequence being inserted
                    self._insert_helper(child, key[prefix_len:], value[prefix_len:])
                    return  

        # the code above helps locate the node to begin insertion from  
        # do insertion at this point
        if len(key):
            if node.parent is None or len(node.value) == self.block_size:  # node is root OR full token block
                # need to insert new leaf node
                # 1. nothing can be matched, insert child node
                # 2. partial matches are with full blocks, e.g. [1,2,3,5] <- [1,2,4,6,7]
                logical_ts = self.logical_ts if self.use_logical_ts else None
                new_node = TreeNode(logical_ts=logical_ts)
                new_node.parent = node
                new_node.key = key[:self.block_size]
                new_node.value = value[:self.block_size]
                node.children[key[:self.block_size]] = new_node
                self._insert_helper(new_node, key[self.block_size:], value[self.block_size:])
                # # motivational experiments
                # self.nodes_created += 1
                # tokens_represented = hash(tuple(new_node.get_all_token_ids()))
                # if tokens_represented not in self.token_representations:
                #     # if True:  # last node [1,2] vs. [1,2,3] are different representations
                #     if len(new_node.key) == self.block_size:  # same representations for the above
                #         self.token_representations.append(tokens_represented)
            else:  # fill the remaining tokens into the current node
                # 3. matched with a partial block, e.g. [1,2,3] <- [1,2,3,4,5]
                assert len(node.children) == 0, f"This node has a partial token block and it should have zero children"
                old_key = node.key
                num_vacant_tokens = self.block_size - len(node.value)
                node.value += value[:num_vacant_tokens]
                node.key += key[:num_vacant_tokens]
                # update node's parent's references to its children
                del node.parent.children[old_key]
                new_key = old_key + key[:num_vacant_tokens]
                node.parent.children[new_key] = node
                self._insert_helper(node, key[num_vacant_tokens:], value[num_vacant_tokens:])

    def _match_prefix_helper(self, node, key, value, nodes_accessed): 
        if not self.use_logical_ts:
            node.last_access_time = time.time()
        else:
            node.last_access_time = self.logical_ts
        
        prefix_stats = []  # list of tuples (shared prefix len, len of child key)
        if len(node.children) != 0:  # node has children. first check if the target node is one of the children.
            for c_key, child in node.children.items():  # c_key: list of token IDs. child: reference to node.
                prefix_stats.append((_key_match(c_key, key), len(c_key),))
            max_prefix_lens = [prefix_len for prefix_len, _ in prefix_stats]
            max_prefix_len = max(max_prefix_lens)
            if max_prefix_len == self.block_size:  # matches with a child that's a full token block, do recursion
                assert max_prefix_lens.count(max_prefix_len) == 1
                child_id = max_prefix_lens.index(max_prefix_len)
                child = list(node.children.values())[child_id]
                value.append(child.value)
                nodes_accessed.append(child)
                return max_prefix_len + self._match_prefix_helper(child, key[max_prefix_len:], value, nodes_accessed)
            elif max_prefix_len != 0:
                # might have a partial match, e.g. [1,2,3] <- [1,2,3,4,5].
                # could also be a case that's not reusable, e.g. [1,2,3,4] <- [1,2,3,5,6]
                for child_id, (prefix_len, child_key_len) in enumerate(prefix_stats):
                    if prefix_len == child_key_len != 0:
                        child = list(node.children.values())[child_id]
                        value.append(child.value)
                        nodes_accessed.append(child)
                        return child_key_len  # no need to keep doing recursion because child is a partial token block 
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
            
    def _print_helper(self, node: TreeNode, depth=0):
        separator = "  "  # "\t"
        for _, child in node.children.items():
            if type(child.hybrid_states) is int:  # for debugging
                hybrid_states_str = str(child.hybrid_states)
            elif type(child.hybrid_states) is HybridStates:
                hybrid_states_str = f"{len(child.hybrid_states.input_ids)} tokens"
            else:
                hybrid_states_str = child.key
            print(separator * depth, "├─", len(child.key), f"last accessed {datetime.fromtimestamp(child.last_access_time).strftime('%I:%M:%S.%f')}, state: {hybrid_states_str}")
            # print(separator * depth, "├─", len(child.key), f"child.key {child.key}, last accessed {datetime.fromtimestamp(child.last_access_time).strftime('%I:%M:%S.%f')}, state: {hybrid_states_str}")
            self._print_helper(child, depth=depth + 1)
        
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
            
        # only evict leaf nodes
        leaves = self._collect_leaves()
        heapq.heapify(leaves)  # leaves is now a min-heap that pops the least recently accessed leaf
        
        bytes_evicted = 0
        while bytes_evicted < bytes_to_remove and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break

            bytes_evicted += self.num_ssm_layers * get_mamba_state_size(self.d, self.n) + self.num_attn_layers * get_kvs_size(len(x.value), self.d)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

