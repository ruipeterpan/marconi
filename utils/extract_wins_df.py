# %%
import re
import itertools
import pandas as pd
import numpy as np

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def extract_configurations(file_path):
    # Open the file and read its content
    with open(file_path, 'r') as file:
        text = file.read()

    texts = text.split("==================================================")

    # Regular expressions to extract the required information
    config_pattern = re.compile(
        r"capacity_bytes ([\d.]+), sessions_per_second ([\d.]+), "
        r"evict_policy_version (\d+), eff_weight ([\d.]+), recency_weight ([\d.]+).*?"
        r"request_hit_rate ([\d.]+)%, token_hit_rate ([\d.]+)%(?:.*?"
        r"Better configuration! token_hit_rate_win ([\d.]+)%, flops_savings_win ([\d.]+)%)?",
        re.DOTALL
    )
    
    # List to store rows for the DataFrame
    data = []
    
    for text in texts:
        # Find all configuration blocks in the text
        matches = config_pattern.findall(text)
        
        # Iterate through each match and append the extracted information to the list
        for match in matches:
            # Convert the matched groups to appropriate data types
            capacity_bytes = float(match[0])
            sessions_per_second = float(match[1])
            evict_policy_version = int(match[2])
            eff_weight = float(match[3])
            recency_weight = float(match[4])
            request_hit_rate = float(match[5])
            token_hit_rate = float(match[6])
            
            # Check if token_hit_rate_win and flops_savings_win are present
            token_hit_rate_win = float(match[7]) if match[7] else np.nan
            flops_savings_win = float(match[8]) if match[8] else np.nan
            
            # if isclose(capacity_bytes, 1e9) and isclose(sessions_per_second, 0.5) and isclose(avg_response_time, 0.25) and isclose(eff_weight, 0.0) and isclose(recency_weight, 0.25):
            #     print("Hello")
            #     print(type(match[9]))
            #     print((match[9]) == "")
            
            if "" not in [match[7], match[8]]:
                # Append the row to the data list, winning configs only
                data.append([
                    capacity_bytes, sessions_per_second,
                    evict_policy_version, eff_weight, recency_weight, request_hit_rate,
                    token_hit_rate, token_hit_rate_win, flops_savings_win
                ])
    
    # Create a DataFrame from the extracted data
    columns = [
        'capacity_bytes', 'sessions_per_second', 
        'evict_policy_version', 'eff_weight', 'recency_weight', 'request_hit_rate',
        'token_hit_rate', 'token_hit_rate_win', 'flops_savings_win'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    return df, text

# %%
file_path = '../sweep/lmsys.txt'
# file_path = '../sweep/sharegpt.txt'
df, text = extract_configurations(file_path)
# print(df)
df["weight_diff"] = df["eff_weight"] - df["recency_weight"]
df = df.sort_values("flops_savings_win", ascending=False)
# df = df.sort_values("token_hit_rate", ascending=False)
print(df.to_string())

# %%
# Define possible values for eff_weight and recency_weight
# eff_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
# recency_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
eff_weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
recency_weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Get all combinations of eff_weight and recency_weight
combinations = list(itertools.product(eff_weights, recency_weights))

# Filter the dataframe for each combination and calculate the average flops_savings
results = []
for eff, rec in combinations:
    subset = df[(df['eff_weight'] == eff) & (df['recency_weight'] == rec)]
    if not subset.empty:
        avg_flops_savings = subset['flops_savings_win'].mean()
        results.append((eff, rec, avg_flops_savings))

# Convert results to dataframe
result_df = pd.DataFrame(results, columns=['eff_weight', 'recency_weight', 'avg_flops_savings'])

result_df

import seaborn as sns
import matplotlib.pyplot as plt

# Corrected pivot statement
heatmap_data = result_df.pivot(index='eff_weight', columns='recency_weight', values='avg_flops_savings')

# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", cbar_kws={'label': 'Avg FLOPS Savings'})
plt.title('Heatmap of Average FLOPS Savings by eff_weight and recency_weight')
plt.xlabel('Recency Weight')
plt.ylabel('Efficiency Weight')
plt.show()


# %%
