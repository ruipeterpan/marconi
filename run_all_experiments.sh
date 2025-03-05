#!/bin/bash

datasets=("sharegpt" "lmsys" "swebench")
# datasets=("lmsys" "swebench")
# datasets=("sharegpt")

# Create logs directory if it doesn't exist
mkdir -p logs

# Loop over each dataset and run the Python script
for dataset in "${datasets[@]}"; do
    echo "Doing a configuration sweep for dataset: $dataset"
    python policy_exploration.py --dataset "$dataset" > "./logs/$dataset.txt" 2>&1
done
