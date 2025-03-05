# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
from radix_cache_hybrid import RadixCache, _key_match

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# Model configuration for NVIDIA's Mamba2-7B Attention-SSM Hybrid LLM
num_ssm_layers = 24
num_attn_layers = 4
num_mlp_layers = 28
d = 4096  # D
n = 128  # N

radix_tree = RadixCache(
    capacity_bytes=1e10,  # bytes
    num_ssm_layers=num_ssm_layers,
    num_attn_layers=num_attn_layers,
    num_mlp_layers=num_mlp_layers,
    d=d,
    n=n,
    evict_policy_version=1,
)

# %%
# Radix tree insertion
for prompt_id, prompt in enumerate([
    "Princeton University is a beautiful place",
    "Princeton University is a good place",
    "Princeton University is a big thing",
    "Princeton is a town",
    "Harvard University is",
    "Princeton University is a big deal",
]):
    print("="*50)
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens.input_ids
    print(f"Prefill prompt ({input_ids.size(1)} tokens):")
    for t in input_ids:
        print(tokenizer.convert_ids_to_tokens(t))
        token_ids = t.tolist()
        print(f"Corresponding token IDs {token_ids}")
        prefix_token_ids, node, branchoff_required, prefix_len = radix_tree.match_prefix(token_ids)
        num_tokens_in_prefix = len(prefix_token_ids)
        print(f"Can reuse {num_tokens_in_prefix} tokens")
        if prefix_len is not None:
            print(f"Identified branchoff node: prefix_len {prefix_len}")
        radix_tree.insert(
            token_ids, 
            state_at_leaf=prompt_id,
            state_at_branchoff=prompt_id,
        )

    radix_tree.pretty_print(verbose=True)

# %%
# Radix tree eviction
radix_tree.evict(bytes_to_remove=1)

# %%
