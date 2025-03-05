# %%
import math
import json
import numpy as np
import collections
import itertools
from tqdm import tqdm
import statistics
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

model_specs = {
    "130m": {
        "d": 768,
    },
    "1.4b": {
        "d": 2048,
    },
    "2.8b": {
        "d": 2560,
    },
    "7b": {
        "d": 4096,
    },
}

model_size = "7b"
d = model_specs[model_size]["d"]
# n = 16
n = 128


def get_attn_layer_flops(l, d=d):
    return 8 * l * d**2 + 4 * l**2 * d

def get_mlp_layer_flops(l, d=d):
    return 16 * l * d**2

def get_mamba1_flops(l, d=d, n=n):
    # return 9 * l * d * n + 2 * l * d**2  # Albert's estimate
    return 12 * l * d**2 + 16 * l * d * n + 10 * l  # Luca's estimate

def get_kvs_size(l, d=d):
    # returns the size in bytes
    # 2 is for k and v; d_model is essentially n_heads * d_head; fp16, 2 bytes/parameter
    return 2 * l * d * 2

def get_mamba_state_size(d=d, n=n, conv_kernel=4, expand=2):
    # ssm states: d_model is essentially n_heads * d_head; n is the hidden recurrent state dimensions; fp16, 2 bytes/parameter
    # conv states: 
    # in_channels = config.intermediate_size + 2 * config.state_size (defaults to 128)
    # intermediate_size is int(expand * self.hidden_size) = 2 * 4096 (for 7B)
    # default conv_kernel=4. conv state size is in_channels * conv_kernel * 2
    intermediate_size = expand * d
    in_channels = intermediate_size + 2 * n
    # return d * n * 2 + in_channels * conv_kernel * 2
    return d * n * 2 + (expand * d + 2 * n) * conv_kernel * 2

def get_model_state_size(l, num_mamba_layers=24, num_attn_layers=4):
    return num_mamba_layers * get_mamba_state_size() + num_attn_layers * get_kvs_size(l)


seqlen = list(range(1, 8192*2))

# state size of a 7b hybrid model
num_mamba_layers = 24
num_attn_layers = 4 
num_mlp_layers = 28

token_block_size = 16
memory_size_hybrid = []
for l in seqlen:
    num_mamba_blocks = math.ceil(l / token_block_size)
    mamba_state_size = get_mamba_state_size(d, n) * num_mamba_layers * num_mamba_blocks
    kvs_size = get_kvs_size(l, d) * num_attn_layers
    memory_size_hybrid.append((mamba_state_size + kvs_size) / 1e6)  # byte -> mb

# 7B transformer
num_attn_layers = 32
memory_size_transformer = []
for l in seqlen:
    kvs_size = get_kvs_size(l, d) * num_attn_layers
    memory_size_transformer.append((kvs_size) / 1e6)  # byte -> mb

num_tokens = 10000
print(f"Hybrid: {num_tokens} tokens {memory_size_hybrid[num_tokens]}")
print(f"Transformer: {num_tokens} tokens {memory_size_transformer[num_tokens]}")

# %%
