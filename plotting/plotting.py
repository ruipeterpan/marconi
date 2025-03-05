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
d = model_specs[model_size]["d"]  # model dimension, or d_model
n = 128  # state dimension, or d_state


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

# attn saved flops per mem: l + 2d
# mamba saved flops per mem: l * (6d/n + 8 + 5/dn)
# Plugging numbers in:
# attn saved flops per mem: l + 8192
# mamba saved flops per mem: l * (192+8) = 200 * l
# they are equivalent when l ~= 50. so when saving a sequence of >50 tokens, saving Mamba states is more effective



# %%
# Fig. 1: state size comparison for 7B models: pure transformer and pure mamba
num_mamba_layers = 64  # 71 MB/token for all layers
num_attn_layers = 32  # 0.52 MB/token for all layers

fig, ax = plt.subplots(figsize=(4.5, 3))
seqlen = list(range(1, 2048))

ax.axhline(y=80*1000, color="red", linestyle="dashed")  # A100-80GB memory
ax.plot(seqlen, [num_attn_layers * get_kvs_size(len) / 1e6 for len in seqlen], label="Transformer", linewidth=2)  # kb -> mb
ax.plot(seqlen, [num_mamba_layers * get_mamba_state_size() / 1e6 for len in seqlen], label="Mamba2 (last token)", linewidth=2)
ax.plot(seqlen, [len * num_mamba_layers * get_mamba_state_size() / 1e6 for len in seqlen], label="Mamba2 (all tokens)", linewidth=2)
ax.text(x=700,y=20*1000, s="Memory of A100-80GB GPU", color="red", fontsize=11)

ax.set_xlabel('Sequence Length', fontsize=11)
ax.set_ylabel('State Size (MB, log-scale)', fontsize=11)
# ax.set_xticks(seqlen)
ax.set_axisbelow(True)  # puts the grid below the bars
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
ax.legend(fontsize=11)
ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# fig.savefig("state_size_comparison.pdf", dpi=500, bbox_inches='tight')


# %% 
# Fig. 2: attn vs. mamba FLOPs *per layer* as seqlen grows

fig, ax = plt.subplots(figsize=(4.5, 3))
# seqlen = list(range(1, 16384))
seqlen = list(range(1, 65536))
# seqlen = list(range(1, 8192))

ax.plot(seqlen, [get_attn_layer_flops(l) for l in seqlen], label="Attn", linewidth=2)
# ax.plot(seqlen, [get_attn_layer_flops(l) + get_mlp_layer_flops(l) for l in seqlen], label="Attn+MLP", linewidth=2)
ax.plot(seqlen, [get_mamba1_flops(l) for l in seqlen], label="Mamba", linewidth=2)

ax.set_title("FLOPs for a single Mamba/Attn block as seqlen scales")
ax.set_xlabel('Sequence length', fontsize=12)
ax.set_ylabel('FLOPs of 1 forward pass (prefill)', fontsize=12)
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
ax.legend(fontsize=12)
plt.show()

# %%
# Fig. 3: FLOPs per memory for Mamba vs. Attn
fig, ax = plt.subplots(figsize=(4.5, 3))
# seqlen = list(range(1, 4096))
seqlen = list(range(1, 512))
# seqlen = list(range(1, 128))

num_mamba_states_saved_for_sequence = 1

ax.plot(seqlen, [get_attn_layer_flops(l) / get_kvs_size(l) for l in seqlen], label="Attn", linewidth=2)
ax.plot(seqlen, [(get_attn_layer_flops(l) + get_mlp_layer_flops(l)) / get_kvs_size(l) for l in seqlen], label="Attn+MLP", linewidth=2)
ax.plot(seqlen, [get_mamba1_flops(l) / (num_mamba_states_saved_for_sequence * get_mamba_state_size()) for l in seqlen], label="Mamba", linewidth=2)
# NOTE(ruipan): this is an inaccurate estimation. FLOPs saved for Transformer should be seqlen - prefix len. or not??

ax.set_title("FLOPs saved per unit of memory")
ax.set_xlabel('Sequence length', fontsize=12)
ax.set_ylabel('FLOPs per byte of memory', fontsize=12)
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
ax.legend(fontsize=12)
plt.show()

# %%
# Fig. 2: attn vs. mamba FLOPs *of all layers layer* as seqlen grows
num_mamba_layers = 24
num_attn_layers = 4 

fig, ax = plt.subplots(figsize=(4.5, 3))
# seqlen = list(range(1, 16384))
# seqlen = list(range(1, 65536))
seqlen = list(range(1, 131072))
# seqlen = list(range(1, 8192))

ax.plot(seqlen, [num_attn_layers * get_attn_layer_flops(l) for l in seqlen], label="Attn", linewidth=2)
ax.plot(seqlen, [num_mamba_layers * get_mamba1_flops(l) for l in seqlen], label="Mamba", linewidth=2)

ax.set_title("FLOPs for all Mamba/Attn layers in a 7B hybrid model as seqlen scales")
ax.set_xlabel('Sequence length', fontsize=12)
ax.set_ylabel('FLOPs of 1 forward pass (prefill)', fontsize=12)
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
ax.legend(fontsize=12)
plt.show()

# %%
# Fig. n: for a hybrid state, how does its flops saving per byte change as seqlen grows
num_mamba_layers = 24
num_attn_layers = 4
num_mlp_layers = 28

fig, ax = plt.subplots(figsize=(4.5, 3))
seqlen = list(range(1, 16384))

y = []
for s in seqlen:
    total_flop_savings = num_mamba_layers * get_mamba1_flops(s, d, n) + \
        num_attn_layers * get_attn_layer_flops(s, d) + \
        num_mlp_layers * get_mlp_layer_flops(s, d)
    total_memory_size = num_mamba_layers * get_mamba_state_size(d, n) + num_attn_layers * get_kvs_size(s, d)
    y.append(total_flop_savings / total_memory_size)

ax.plot(seqlen, y, linewidth=2)

ax.set_title("FLOPs per byte for the state of a 7B hybrid model as seqlen scales")
ax.set_xlabel('Sequence length', fontsize=12)
ax.set_ylabel('FLOPs savings per byte', fontsize=12)
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
# ax.legend(fontsize=12)
plt.show()

# %%
# Fig. n: State size of a hybrid model as the seqlen scales
num_mamba_layers = 24
num_attn_layers = 4 

fig, ax = plt.subplots(figsize=(4.5, 3))
# seqlen = list(range(1, 16384))
# seqlen = list(range(1, 65536))
# seqlen = list(range(1, 131072))
seqlen = list(range(1, 1024))

attn = [num_attn_layers * get_kvs_size(l) / 1e6 for l in seqlen]
mamba = [num_mamba_layers * get_mamba_state_size() / 1e6 for l in seqlen]
# y = np.vstack([attn, mamba])
y = np.vstack([mamba, attn])
# plt.stackplot(seqlen, y, labels=['4 Attn layers', '24 Mamba layers'], colors=['#ff9999', '#66b3ff'])
# plt.stackplot(seqlen, y, labels=['4 Attn layers', '24 Mamba layers'])
plt.stackplot(seqlen, y, labels=['24 Mamba layers', '4 Attn layers'])

# ax.plot(seqlen, [num_attn_layers * get_kvs_size(l) / 1e6 for l in seqlen], label="4 Attn layers", linewidth=2)
# ax.plot(seqlen, [num_mamba_layers * get_mamba_state_size() / 1e6 for l in seqlen], label="24 Mamba layers", linewidth=2)
# ax.plot(seqlen, [(num_attn_layers * get_kvs_size(l) + num_mamba_layers * get_mamba_state_size()) / 1e6 for l in seqlen], label="Total", linewidth=2)

ax.set_title("State size of a 7B hybrid model as seqlen scales")
ax.set_xlabel('Num Tokens', fontsize=12)
ax.set_ylabel('State Size (MB)', fontsize=12)
ax.ticklabel_format(style = 'plain')
ax.set_axisbelow(True)  # puts the grid below the bars
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
ax.legend(fontsize=12)
plt.show()


# %%
# Fig. n: FLOPs of different layers of a hybrid model as the seqlen scales
num_mamba_layers = 24
num_attn_layers = 4 
num_mlp_layers = 28

# fig, ax = plt.subplots(figsize=(4.5, 3))
fig, ax = plt.subplots(figsize=(8, 3))
seqlen = list(range(1, 16384))
# seqlen = list(range(1, 65536))
# seqlen = list(range(1, 131072))
# seqlen = list(range(1, 1024))

attn = [num_attn_layers * get_attn_layer_flops(l) for l in seqlen]
mamba = [num_mamba_layers * get_mamba1_flops(l) for l in seqlen]
mlp = [num_mlp_layers * get_mlp_layer_flops(l) for l in seqlen]
# y = np.vstack([attn, mamba])
y = np.vstack([mamba, attn, mlp])
# plt.stackplot(seqlen, y, labels=['4 Attn layers', '24 Mamba layers'], colors=['#ff9999', '#66b3ff'])
# plt.stackplot(seqlen, y, labels=['4 Attn layers', '24 Mamba layers'])
plt.stackplot(seqlen, y, labels=['24 Mamba layers', '4 Attn layers', '28 MLP layers'])

# ax.plot(seqlen, [num_attn_layers * get_kvs_size(l) / 1e6 for l in seqlen], label="4 Attn layers", linewidth=2)
# ax.plot(seqlen, [num_mamba_layers * get_mamba_state_size() / 1e6 for l in seqlen], label="24 Mamba layers", linewidth=2)
# ax.plot(seqlen, [(num_attn_layers * get_kvs_size(l) + num_mamba_layers * get_mamba_state_size()) / 1e6 for l in seqlen], label="Total", linewidth=2)

ax.set_title("FLOPs of a 7B hybrid model as seqlen scales")
ax.set_xlabel('Num Tokens', fontsize=12)
ax.set_ylabel('FLOPs', fontsize=12)
# ax.ticklabel_format(style = 'plain')
ax.set_axisbelow(True)  # puts the grid below the bars
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
ax.legend(fontsize=12)
plt.show()




# %%
# Fig. n: how does the flops saving per byte change as layer ratio changes

fig, ax = plt.subplots(figsize=(2.5 * 1.6, 2.5))
# linecolors = ["#2D6A4F", "#52B788", "#95D5B2"]
linecolors = ['#52B788', '#40916C', '#2D6A4F', "#081C15"]
linestyles = ["solid", "dotted", "dashed"]
fontsize = 12

# seqlen = list(range(1, 16384))
# seqlen = list(range(1, 32000))
# seqlen = list(range(1, 65536))
# seqlen = list(range(1, 131072))
# seqlen = list(range(1, 1024))
# seqlen = list(range(1, 4096))
# seqlen = list(range(1, 256))
# seqlen = list(range(1, 1024))
seqlen = list(range(1, 2048))

# # mamba 7b
num_mamba_layers = 64
y = []
for s in seqlen:
    total_flop_savings = num_mamba_layers * get_mamba1_flops(s, d, n)
    total_memory_size = num_mamba_layers * get_mamba_state_size(d, n)
    y.append(total_flop_savings / total_memory_size)
ax.plot(seqlen, y, linewidth=2, label="Mamba", color=linecolors[0], linestyle=linestyles[0])

# # mamba 7b hybrid
num_mamba_layers = 24
num_attn_layers = 4
num_mlp_layers = 28
y = []
for s in seqlen:
    total_flop_savings = num_mamba_layers * get_mamba1_flops(s, d, n) + \
        num_attn_layers * get_attn_layer_flops(s, d) + \
        num_mlp_layers * get_mlp_layer_flops(s, d)
    total_memory_size = num_mamba_layers * get_mamba_state_size(d, n) + num_attn_layers * get_kvs_size(s, d)
    y.append(total_flop_savings / total_memory_size)
ax.plot(seqlen, y, linewidth=2, label="Hybrid", color=linecolors[1], linestyle=linestyles[1])

# # llama 7b
num_attn_layers = 32
num_mlp_layers = 32
y = []
for s in seqlen:
    total_flop_savings = num_attn_layers * get_attn_layer_flops(s, d) + \
        num_mlp_layers * get_mlp_layer_flops(s, d)
    total_memory_size = num_attn_layers * get_kvs_size(s, d)
    y.append(total_flop_savings / total_memory_size)
ax.plot(seqlen, y, linewidth=2, label="Transformer", color=linecolors[2], linestyle=linestyles[2])

# ax.set_title("FLOPs per byte for different 7B models as seqlen scales")
ax.set_xlabel('Sequence Length', fontsize=fontsize)
ax.set_ylabel('FLOP Saved/Byte', fontsize=fontsize)
ax.ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
ax.legend(fontsize=fontsize)
plt.show()
fig.savefig("./figures/flop_efficiency_diff.pdf", dpi=500, bbox_inches='tight')



# %%
# Fig. n: state size comparison of different checkpoint frequencies for a 7B hybrid model
num_mamba_layers = 24
num_attn_layers = 4 
num_mlp_layers = 28

# num_mamba_layers = 0
# num_attn_layers = 32
# num_mlp_layers = 32

fig, ax = plt.subplots(figsize=(2.5 * 1.6, 2.5))
seqlen = list(range(1, 8192*2))
# linecolors = ["black", "tab:orange", "tab:green"]
linecolors = ["#2D6A4F", "#52B788", "#95D5B2"]
fontsize = 12

token_block_sizes = [8, 16, 32]
token_id = 10000
for token_block_size in token_block_sizes:
    memory_size = []
    for l in seqlen:
        num_mamba_blocks = math.ceil(l / token_block_size)
        mamba_state_size = get_mamba_state_size(d, n) * num_mamba_layers * num_mamba_blocks
        kvs_size = get_kvs_size(l, d) * num_attn_layers
        memory_size.append((mamba_state_size + kvs_size) / 1e6)  # byte -> mb
    ax.plot(seqlen, memory_size, label=f"block_size={token_block_size}", linewidth=2, color=linecolors.pop(0))
    print(f"Block size {token_block_size}, at {token_id}, size is {memory_size[token_id]} MB")
ax.axhline(y=40*1000, color="black", linestyle="dashed")  # A100-80GB memory
ax.text(x=700,y=16*1000, s="40 GB", color="black", fontsize=fontsize)

ax.set_xlabel('Sequence Length', fontsize=fontsize)
ax.set_ylabel('State Size\n(MB, log-scale)', fontsize=fontsize)
# ax.set_xticks(seqlen)
ax.set_axisbelow(True)  # puts the grid below the bars
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
ax.legend(fontsize=fontsize)
ax.tick_params(axis='both', labelsize=fontsize)
ax.set_yscale("log")
# ax.set_xscale("log")
plt.show()
# fig.savefig("./figures/vllm_memory_overhead.pdf", dpi=500, bbox_inches='tight')


# %%
# Input/output length distribution
def get_trace_path(config):
    trace_name, num_sessions, capacity_bytes, sessions_per_second, avg_response_time = config
    trace_filename = f"./traces/{trace_name}_sps={sessions_per_second}"
    if trace_name in ["swebench"]:
        trace_filename += f"_art={avg_response_time}"
    trace_filename += f"_nums={num_sessions}.jsonl"
    return trace_filename

def analyze_trace(trace_path):
    with open(trace_path, 'r') as json_file:
        json_list = list(json_file)

        trace = []
        for json_str in tqdm(json_list):
            result = json.loads(json_str)
            trace.append(result)
    
    print(f"Num requests: {len(trace)}")

    # input/output length distribution
    input_tokens = [x["num_input_tokens"] for x in trace]
    output_tokens = [x["num_output_tokens"] for x in trace]
    
    return input_tokens, output_tokens

fig, axs = plt.subplots(1, 3, layout='constrained', figsize=(7, 2.2))
colors = ["#2D6A4F", "#52B788", "#95D5B2"]
# bins = 25
fontsize = 12
configs = [  # (trace_name, num_sessions, capacity_bytes, sessions_per_second, avg_response_time)
    ("lmsys", 100, 2e9, 1, None,),
    ("sharegpt", 100, 1e9, 5, None,),
    ("swebench", 100, 1e10, 0.25, 5,),
]

for i, config in enumerate(configs):
    trace_path = get_trace_path(config)
    input_tokens, output_tokens = analyze_trace(trace_path)
    
    w = [500, 100, 500][i]  # bin width
    axs[i].hist(input_tokens, label="Input", color=colors[0], bins=np.arange(min(input_tokens), max(input_tokens) + w, w))  # bins=bins, 
    axs[i].hist(output_tokens, label="Output", color=colors[1], bins=np.arange(min(output_tokens), max(output_tokens) + w, w))  # bins=bins, 
    
    axs[i].set_xlabel("# Tokens", fontsize=fontsize)
    if i == 0:
        axs[i].set_ylabel("Density", fontsize=fontsize)
    axs[i].ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
    # axs[i].legend()
    title = [
        "(a) LMSys", "(b) ShareGPT", "(c) SWEBench"
    ]
    axs[i].set_title(title[i], y=-0.6, fontsize=fontsize+3)
    if i == 1:
        axs[i].legend(loc="upper center", ncols=2, fontsize=fontsize, bbox_to_anchor=(0.5, 1.3), handlelength=.8, frameon=False, borderaxespad=0)  # , mode="expand"
    print(f"scheme {title[i]}, max output_tokens {max(output_tokens)}, avg output_tokens {statistics.mean(output_tokens)}")
plt.show()
fig.savefig("./figures/input_len_distribution.pdf", dpi=500, bbox_inches='tight')

# %%
# Fig. n: FLOPs of different layers of a hybrid model as the seqlen scales
num_mamba_layers = 24
num_attn_layers = 4 
num_mlp_layers = 28

# fig, ax = plt.subplots(figsize=(4.5, 3))
fig, ax = plt.subplots(figsize=(8, 3))
# seqlen = list(range(1, 16384))
# seqlen = list(range(1, 32000))
seqlen = list(range(1, 65536))
# seqlen = list(range(1, 131072))
# seqlen = list(range(1, 1024))
colors = ["#2D6A4F", "#52B788", "#95D5B2"]
fontsize = 12

attn = [num_attn_layers * get_attn_layer_flops(l) for l in seqlen]
mamba = [num_mamba_layers * get_mamba1_flops(l) for l in seqlen]
mlp = [num_mlp_layers * get_mlp_layer_flops(l) for l in seqlen]
# y = np.vstack([attn, mamba])
y = np.vstack([mamba, attn, mlp])
# plt.stackplot(seqlen, y, labels=['4 Attn layers', '24 Mamba layers'], colors=['#ff9999', '#66b3ff'])
# plt.stackplot(seqlen, y, labels=['4 Attn layers', '24 Mamba layers'])
plt.stackplot(seqlen, y, colors=colors, labels=['24 SSM layers', '4 Attn layers', '28 MLP layers'])

# ax.plot(seqlen, [num_attn_layers * get_kvs_size(l) / 1e6 for l in seqlen], label="4 Attn layers", linewidth=2)
# ax.plot(seqlen, [num_mamba_layers * get_mamba_state_size() / 1e6 for l in seqlen], label="24 Mamba layers", linewidth=2)
# ax.plot(seqlen, [(num_attn_layers * get_kvs_size(l) + num_mamba_layers * get_mamba_state_size()) / 1e6 for l in seqlen], label="Total", linewidth=2)

# ax.set_title("FLOPs of a 7B hybrid model as seqlen scales")
ax.set_xlabel('Sequence Length', fontsize=fontsize)
ax.set_ylabel('FLOP', fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.yaxis.get_offset_text().set_fontsize(fontsize)
# ax.ticklabel_format(style = 'plain')
ax.set_axisbelow(True)  # puts the grid below the bars
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
ax.legend(fontsize=fontsize)
plt.show()
# fig.savefig("./figures/flop_breakdown.pdf", dpi=500, bbox_inches='tight')

# %%
# # How much FLOPs does caching Mamba and Attn states save, respectively
# seq_lens = [8000]  # 1000, 2000, 3000, 4000, 5000, 6000, 
# prefix_lens = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]

# for seqlen, prefix_len in itertools.product(seq_lens, prefix_lens):
#     if prefix_len < seqlen:
#         mamba_flops_saved = get_mamba1_flops(seqlen) - get_mamba1_flops(seqlen - prefix_len)
#         # mamba_flops_saved = get_mamba_state_size(prefix_len)
#         # FLOPs saved by attention: should probably be just get_attn_layer_flops(prefix_len), 
#         # because non-prefix attn operation still needs to attend to the prefix
#         # attn_flops_saved = get_attn_layer_flops(seqlen) - get_attn_layer_flops(seqlen - prefix_len)
#         attn_flops_saved = get_attn_layer_flops(prefix_len)
        
#         print(f"seqlen {seqlen}, prefix_len {prefix_len}, mamba_flops_saved {mamba_flops_saved / 1e12}e12, attn_flops_saved {attn_flops_saved / 1e12}e12")
#         print(f"mamba/attn {mamba_flops_saved / attn_flops_saved}")
# %%
num_mamba_layers = 64  # 71 MB/token for all layers
num_attn_layers = 32  # 0.52 MB/token for all layers
colors = ["#2D6A4F", "#52B788", "#95D5B2"]
linestyles = ["solid", "dotted", "dashed"]
fontsize = 13

fig, ax = plt.subplots(figsize=(3, 2.7), layout="constrained")
seqlen = list(range(1, 256))

ax.plot(seqlen, [get_kvs_size(len) / 1e6 for len in seqlen], label="Attention", color=colors[1], linestyle=linestyles[2], linewidth=2)  # kb -> mb
ax.plot(seqlen, [get_mamba_state_size() / 1e6 for len in seqlen], label="SSM", color=colors[0], linestyle=linestyles[0], linewidth=2)
# ax.plot(seqlen, [len * num_mamba_layers * get_mamba_state_size() / 1e6 for len in seqlen], label="Mamba2 (all tokens)", linewidth=2)

ax.set_xlabel('Sequence Length', fontsize=fontsize)
ax.set_ylabel('Layer State Size (MB)', fontsize=fontsize)  # , log-scale
ax.tick_params(axis='both', which='major', labelsize=fontsize)
# ax.set_xticks(seqlen)
ax.set_axisbelow(True)  # puts the grid below the bars
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
ax.legend(fontsize=fontsize)
# ax.set_yscale("log")
# ax.set_xscale("log")
fig.savefig("./figures/state_size_comparison.pdf", dpi=500, bbox_inches='tight')
# %%
