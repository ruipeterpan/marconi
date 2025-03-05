# %%
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
tqdm.pandas()

device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)

# %%
# def contains_unicode(s):
#     # Check if any character in the string has a Unicode code point greater than 127 (non-ASCII)
#     return any(ord(char) > 127 for char in s)

def contains_many_unicode(s):
    # Check if any character in the string has a Unicode code point greater than 127 (non-ASCII)
    is_unicode_list = [ord(char) > 127 for char in s]
    return (sum(is_unicode_list) / len(is_unicode_list)) > 0.5


def get_num_tokens(x):
    # x is a numpy.ndarray
    num_rounds = len(x)
    full_history = "".join([json.dumps(x[i]) for i in range(num_rounds)])
    tokens = tokenizer(x, return_tensors="pt").to(device)
    input_ids = tokens.input_ids
    return input_ids.size(1)

def process_lmsys_dataset(
    min_num_rounds: int = 10,
):
    ds = load_dataset("lmsys/lmsys-chat-1m")
    # convert to pandas
    ds.set_format(type="pandas")
    df = ds["train"][:]
    df = df[df["turn"] >= min_num_rounds]  # only keep conversations with 10+ turns

    df["conversation_str"] = df["conversation"].progress_apply(lambda x: str(x))
    df["num_tokens"] = df["conversation_str"].progress_apply(get_num_tokens)

    df.to_pickle("lmsys_chat_1m.pickle")



def generate_lmsys_trace(
    sessions_per_second: float,
    num_sessions: int = None,  # only generate traces for the first n sessions
    words_per_min: float = 90.0,  # words per minute, Rui's typing speed
):
    words_per_second = words_per_min / 60.0  # words per second
    
    df = pd.read_pickle("../datasets/lmsys_chat_1m.pickle")
    # drop sessions that has many unicode characters
    df["contains_many_unicode"] = df["conversation_str"].apply(lambda x: contains_many_unicode(x))
    df = df[df["contains_many_unicode"] == False]
    df.reset_index()
    
    if num_sessions is None:
        num_sessions = len(df.index)
    all_requests = []
    
    for session_id in tqdm(range(num_sessions)):
        num_turns = df.iloc[session_id]["turn"]
        assert num_turns * 2 == len(df.iloc[session_id]["conversation"])
        
        curr_ts = session_id / sessions_per_second  # timestamp in second: the ts of the first request in this session
        requests = []
        
        conv_history_ids = []  # token IDs of the conversation history
        
        for turn_id in range(num_turns):
            user_input = json.dumps(df.iloc[session_id]["conversation"][2 * turn_id])
            llm_output = json.dumps(df.iloc[session_id]["conversation"][2 * turn_id + 1])
            user_input_content = json.dumps(df.iloc[session_id]["conversation"][2 * turn_id]["content"])
            
            # print(f"Turn {turn_id},\n\tuser_input {user_input},\n\tllm_output {llm_output}")

            tokens = tokenizer(user_input, return_tensors="pt")
            user_input_tokens = tokens.input_ids[0].tolist()
            
            tokens = tokenizer(llm_output, return_tensors="pt")
            llm_output_tokens = tokens.input_ids[0].tolist()  # weird issue: the last token of input_tokens is different from the token at the same index in output_tokens.
            
            if len(conv_history_ids + user_input_tokens + llm_output_tokens) > 32768:
                # skip all requests with >32k input tokens
                break
            
            if turn_id != 0:
                # Inter-request latency is user's typing speed
                num_input_words = user_input_content.count(' ')
                typing_latency = num_input_words / words_per_second
                curr_ts += typing_latency
                # print(f"Session {session_id}, turn {turn_id}, {num_input_words} input words, typing_latency {typing_latency}. user_input_content {user_input_content}")
                
                # curr_ts += np.random.poisson(lam=avg_response_time, size=1)[0]  # or, take from Poisson distribution
                
            requests.append({
                "session_id": session_id,
                "turn_id": turn_id,
                "ts": curr_ts,
                "num_input_tokens": len(conv_history_ids + user_input_tokens),
                "num_output_tokens": len(llm_output_tokens),
                "input_tokens": conv_history_ids + user_input_tokens,
                "output_tokens": llm_output_tokens,  # not including the input tokens
            })
            
            conv_history_ids += (user_input_tokens + llm_output_tokens)
        
        all_requests += requests
        
    all_requests = sorted(all_requests, key=lambda x: x["ts"])
    print(f"Generated {len(all_requests)} requests")
    
    with open(f"../traces/lmsys_sps={sessions_per_second}_nums={num_sessions}.jsonl", 'w') as f:
        for r in all_requests:
            json.dump(r, f)
            f.write('\n')
    
    return all_requests, df


def process_sharegpt_dataset(
    min_num_rounds: int = 10,
):
    # filename = "../datasets/sg_90k_part1.json"  # https://huggingface.co/datasets/RyokoAI/ShareGPT52K
    filename = "../datasets/ShareGPT_V3_unfiltered_cleaned_split.json"  # https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    
    with open(filename) as f:
        dataset = json.load(f)
    
    # filter out short conversations
    # num_turns * 2 == num_messages
    dataset = [d for d in dataset if len(d["conversations"]) >= min_num_rounds * 2]
    
    num_conversations = len(dataset)
    print(f"After filtering, dataset contains {num_conversations} sessions")
    return dataset
    
    


def generate_sharegpt_trace(
    sessions_per_second: float,
    num_sessions: int = None,  # only generate traces for the first n sessions
    words_per_min: float = 90.0,  # words per minute, Rui's typing speed
):
    words_per_second = words_per_min / 60.0  # words per second
    
    dataset = process_sharegpt_dataset()
    
    if num_sessions is None:
        num_sessions = len(dataset)
    all_requests = []
    
    for session_id in tqdm(range(num_sessions)):
        num_turns = int(len(dataset[session_id]["conversations"]) / 2)
        
        curr_ts = session_id / sessions_per_second  # timestamp in second: the ts of the first request in this session
        requests = []
        
        conv_history_ids = []  # token IDs of the conversation history
        
        for turn_id in range(num_turns):
            user_input = json.dumps(dataset[session_id]["conversations"][2 * turn_id])
            llm_output = json.dumps(dataset[session_id]["conversations"][2 * turn_id + 1])
            user_input_content = dataset[session_id]["conversations"][2 * turn_id]["value"]
            
            print(f"Turn {turn_id},\n\tuser_input {user_input},\n\tllm_output {llm_output}")

            tokens = tokenizer(user_input, return_tensors="pt")
            user_input_tokens = tokens.input_ids[0].tolist()
            
            tokens = tokenizer(llm_output, return_tensors="pt")
            llm_output_tokens = tokens.input_ids[0].tolist()  # weird issue: the last token of input_tokens is different from the token at the same index in output_tokens.
            
            if len(conv_history_ids + user_input_tokens) > 32768:
                # skip all requests with >32k input tokens
                break
            
            if turn_id != 0:
                # Inter-request latency is user's typing speed
                num_input_words = user_input_content.count(' ')
                typing_latency = num_input_words / words_per_second
                curr_ts += typing_latency
                # print(f"Session {session_id}, turn {turn_id}, {num_input_words} input words, typing_latency {typing_latency}. user_input_content {user_input_content}")
                
                # curr_ts += np.random.poisson(lam=avg_response_time, size=1)[0]  # or, take from Poisson distribution
                
            requests.append({
                "session_id": session_id,
                "turn_id": turn_id,
                "ts": curr_ts,
                "num_input_tokens": len(conv_history_ids + user_input_tokens),
                "num_output_tokens": len(llm_output_tokens),
                "input_tokens": conv_history_ids + user_input_tokens,
                "output_tokens": llm_output_tokens,  # not including the input tokens
            })
            
            conv_history_ids += (user_input_tokens + llm_output_tokens)
        
        all_requests += requests
        
    all_requests = sorted(all_requests, key=lambda x: x["ts"])
    print(f"Generated {len(all_requests)} requests")
    num_input_tokens = [x["num_input_tokens"] for x in all_requests]
    import statistics
    print(f"num_input_tokens: max {max(num_input_tokens)}, min {min(num_input_tokens)}, mean {statistics.mean(num_input_tokens)}")
    
    with open(f"../traces/sharegpt_sps={sessions_per_second}_nums={num_sessions}.jsonl", 'w') as f:
        for r in all_requests:
            json.dump(r, f)
            f.write('\n')
    
    return all_requests




def process_swebench_trace(
    sessions_per_second: float,  # arrival rate of sessions
    avg_response_time: int,  # average time (s) for the agent to execute actions
    seed: int = 42,
    num_sessions: int = 100,
    collapse_except_last: int = 5,  # observations preceding the last 5 are each collapsed into a single line -- following the guidelines in [SWEAgent](https://arxiv.org/pdf/2405.15793)
):
    np.random.seed(seed)
    model_name = "20240820_honeycomb"
    dirname = f"/home/ubuntu/SWE-bench-experiments/evaluation/verified/{model_name}/trajs"
    trace_name_list = sorted(os.listdir(dirname))
    
    if num_sessions is None:
        num_sessions = len(trace_name_list)
    all_requests = []
    
    # for session_id, trace_name in enumerate(tqdm(trace_name_list)):
    for session_id in tqdm(range(num_sessions)):
        trace_name = trace_name_list[session_id]
        with open(os.path.join(dirname, trace_name), "r") as f:
            traj = json.load(f)

        num_messages = len(traj)
        num_turns = int(num_messages / 2)

        curr_ts = session_id / sessions_per_second
        requests = []
        conv_history_ids_list = []  # token IDs of the conversation history
        # conv_history_ids is a list of lists of lists:
        # [[round 1 user_input, round 1 llm_output], [round 2 user_input, round 2 llm_output], ...]

        for turn_id in range(num_turns):
            conv_history_ids = [item for sublist1 in conv_history_ids_list for sublist2 in sublist1 for item in sublist2]
            user_input = json.dumps(traj[2 * turn_id])
            llm_output = json.dumps(traj[2 * turn_id + 1])
            
            # print(f"Turn {turn_id},\n\tuser_input {user_input},\n\tllm_output {llm_output}")

            tokens = tokenizer(user_input, return_tensors="pt")
            user_input_tokens = tokens.input_ids[0].tolist()
            
            tokens = tokenizer(llm_output, return_tensors="pt")
            llm_output_tokens = tokens.input_ids[0].tolist()  # weird issue: the last token of input_tokens is different from the token at the same index in output_tokens.
            
            if turn_id != 0:
                curr_ts += np.random.poisson(lam=avg_response_time, size=1)[0]  # or, take from Poisson distribution
            
            # if len(conv_history_ids + user_input_tokens) > 32768:
            if len(conv_history_ids + user_input_tokens) > 32768 or turn_id > 50:
                # skip all requests with >32k input tokens or exceeds 50 rounds
                break
            
            requests.append({
                "session_id": session_id,
                "turn_id": turn_id,
                "ts": curr_ts,
                "num_input_tokens": len(conv_history_ids + user_input_tokens),
                "num_output_tokens": len(llm_output_tokens),
                "input_tokens": conv_history_ids + user_input_tokens,
                "output_tokens": llm_output_tokens,  # not including the input tokens
            })
            
            # print(f"session {session_id}, turn {turn_id}, num_input_tokens {len(conv_history_ids + user_input_tokens)}")
            
            # conv_history_ids += (user_input_tokens + llm_output_tokens)
            # print(f"Num full interaction pairs is {[len(x) for x in conv_history_ids_list]}")
            if [len(x) for x in conv_history_ids_list].count(2) >= collapse_except_last + 1:
                conv_history_ids_list[-(collapse_except_last)].pop(0)  # collapse the environment observation
                # print(f"popping observation of round {-(collapse_except_last)}")
            conv_history_ids_list.append([user_input_tokens, llm_output_tokens])


        all_requests += requests
    
    all_requests = sorted(all_requests, key=lambda x: x["ts"])
    print(f"Generated {len(all_requests)} requests")
    
    with open(f"../traces/swebench_sps={sessions_per_second}_art={avg_response_time}_nums={num_sessions}.jsonl", 'w') as f:
    # with open(f"../traces/archive/SWEBench_manyrounds_collapsed/swebench_sps={sessions_per_second}_art={avg_response_time}_nums={num_sessions}.jsonl", 'w') as f:
        for r in all_requests:
            json.dump(r, f)
            f.write('\n')
    
    return all_requests
    
    
def process_wildchat_dataset(
    min_num_rounds: int = 10,
):
    ds = load_dataset("allenai/WildChat-1M")
    # convert to pandas
    ds.set_format(type="pandas")
    df = ds["train"][:]
    df = df[df["turn"] >= min_num_rounds]  # only keep conversations with 10+ turns
    df = df[df["language"] == "English"]  # only evaluate on English conversations
    
    df["conversation_str"] = df["conversation"].progress_apply(lambda x: str(x))
    df["num_tokens"] = df["conversation_str"].progress_apply(get_num_tokens)

    df.to_pickle("../datasets/wildchat_1m.pickle")
    
    return df

def generate_wildchat_trace(
    sessions_per_second: float,
    num_sessions: int = None,  # only generate traces for the first n sessions
    words_per_min: float = 90.0,  # words per minute, Rui's typing speed
):
    words_per_second = words_per_min / 60.0  # words per second
    
    df = pd.read_pickle("../datasets/wildchat_1m.pickle")
    # drop sessions that has many unicode characters
    df["contains_many_unicode"] = df["conversation_str"].apply(lambda x: contains_many_unicode(x))
    df = df[df["contains_many_unicode"] == False]
    df.reset_index()
    
    if num_sessions is None:
        num_sessions = len(df.index)
    all_requests = []
    
    for session_id in tqdm(range(num_sessions)):
        num_turns = df.iloc[session_id]["turn"]
        assert num_turns * 2 == len(df.iloc[session_id]["conversation"])
        
        curr_ts = session_id / sessions_per_second  # timestamp in second: the ts of the first request in this session
        requests = []
        
        conv_history_ids = []  # token IDs of the conversation history
        
        for turn_id in range(num_turns):
            user_input_dict = df.iloc[session_id]["conversation"][2 * turn_id]
            llm_output_dict = df.iloc[session_id]["conversation"][2 * turn_id + 1]
            # remove unnecessary keys: timestamp is not json serializable, location info not necessary, etc.
            for key_to_remove in ["country", "hashed_ip", "header", "language", "redacted", "state", "timestamp", "toxic", "turn_identifier"]:
                user_input_dict.pop(key_to_remove, None)
                llm_output_dict.pop(key_to_remove, None)
            user_input = json.dumps(user_input_dict)
            llm_output = json.dumps(llm_output_dict)
            user_input_content = json.dumps(df.iloc[session_id]["conversation"][2 * turn_id]["content"])
            
            # print(f"Turn {turn_id},\n\tuser_input {user_input},\n\tllm_output {llm_output}")

            tokens = tokenizer(user_input, return_tensors="pt")
            user_input_tokens = tokens.input_ids[0].tolist()
            
            tokens = tokenizer(llm_output, return_tensors="pt")
            llm_output_tokens = tokens.input_ids[0].tolist()  # weird issue: the last token of input_tokens is different from the token at the same index in output_tokens.
            
            if turn_id != 0:
                # Inter-request latency is user's typing speed
                num_input_words = user_input_content.count(' ')
                typing_latency = num_input_words / words_per_second
                curr_ts += typing_latency
                # print(f"Session {session_id}, turn {turn_id}, {num_input_words} input words, typing_latency {typing_latency}. user_input_content {user_input_content}")
                
                # curr_ts += np.random.poisson(lam=avg_response_time, size=1)[0]  # or, take from Poisson distribution
                
            requests.append({
                "session_id": session_id,
                "turn_id": turn_id,
                "ts": curr_ts,
                "num_input_tokens": len(conv_history_ids + user_input_tokens),
                "num_output_tokens": len(llm_output_tokens),
                "input_tokens": conv_history_ids + user_input_tokens,
                "output_tokens": llm_output_tokens,  # not including the input tokens
            })
            
            conv_history_ids += (user_input_tokens + llm_output_tokens)
        
        all_requests += requests
        
    all_requests = sorted(all_requests, key=lambda x: x["ts"])
    print(f"Generated {len(all_requests)} requests")
    
    with open(f"../traces/wildchat_sps={sessions_per_second}_nums={num_sessions}.jsonl", 'w') as f:
        for r in all_requests:
            json.dump(r, f)
            f.write('\n')
    
    return all_requests, df

# %%
# all_requests = process_swebench_trace(
#     sessions_per_second=1,
#     avg_response_time=2,
#     num_sessions=500,
# )

# %%
for sps in [0.25, 0.5, 1, 2, 5, 10]:
    for avg_response_time in [5, 7.5, 10]:
        all_requests = process_swebench_trace(
            sessions_per_second=sps,
            avg_response_time=avg_response_time,
            num_sessions=100,
        )
        
# %%
for sps in [0.25, 0.5, 1, 2, 5, 10]:
    all_requests, df = generate_lmsys_trace(sps, num_sessions=100)

# %%
for sps in [0.25, 0.5, 1, 2, 5, 10]:
    all_requests = generate_sharegpt_trace(sps, num_sessions=100)


# %%
