# MLSys 2025 Experiments

## Abstract

In this artifact, we describe how to reproduce all experiments in this paper.
The evaluation utilizes request arrival traces from LMSys, ShareGPT, and SWEBench, all tokenized using the `meta-llama/Llama-2-7b-hf` tokenizer for consistency.
Key experiments involve running vLLM+, eviction policies V1 (SGLang+), V2 (Marconi), and V3 (offline-optimal, static-α oracle policy) across various dataset/arrival rate/cache size combinations, 
and reproducing figures using plotting scripts in the `/plotting` directory.
This artifact supports easy customizations, including implementing additional eviction policies and evaluating new datasets and model configurations.

## Artifact check-list (meta-information)

- Program: Python
- Disk space required: ~20 GB
- Time needed to prepare workflow: ~1 hour
- Time needed to complete experiments: ~12 hours
- Publicly available: Yes
- Code licenses: [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/)
- Archived DOI: [10.5281/zenodo.14970139](10.5281/zenodo.14970139)

## Description

### How delivered

The artifact may be downloaded from zenodo at [https://zenodo.org/records/14970139](https://zenodo.org/records/14970139), or cloned from the GitHub repository at [https://github.com/ruipeterpan/marconi](https://github.com/ruipeterpan/marconi). The request arrival traces (with token-level information) can be downloaded from Google Drive by following the documentation below.

### Hardware dependencies

We have tested Marconi on Cloudlab nodes with Ubuntu 22.04 and Python 3.11.9 (in theory, our source code should be runnable on any Linux machine). 
Due to the request arrival traces containing token IDs for each request, they require a total disk size of ~7 GB to house.

> :wave: Artifact evaluators: Hello! If you want access to Cloudlab, please send your public key to ruipan@princeton.edu and we will set up a Cloudlab node for you to reproduce our experiments.

### Software dependencies

For ease of reproduction, we use conda to create a virtual environment (Miniconda can be installed by following the instructions 
in [this doc](https://www.anaconda.com/docs/getting-started/miniconda/install#basic-install-instructions)).

We have prepared an `environment.yml` file that lists the dependencies and the versions of the dependencies. Once conda has been installed, 
create an environment from the .yml file by following the instructions in 
[this doc](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

```bash
# clone this repo
git clone https://github.com/ruipeterpan/marconi.git
conda env create -f ./marconi/environment.yml
conda activate marconi
```

### Datasets

Marconi's evaluation is done on a sweep of request arrival traces crafted from LMSys, ShareGPT, and SWEBench. For ease and efficiency of reproduction, 
we provide all the request arrival traces used in this paper. We used `meta-llama/Llama-2-7b-hf` to tokenize all the requests to ensure consistency 
when evaluating the same trace on different model architectures. Due to the time to tokenize requests, we compress them (~700M/6.3G pre/post compression) 
and host them on Google Drive and they can be downloaded via `gdown`. Alternatively, they can be generated from raw datasets using utility scripts in `/utils`. 
The raw datasets were originally downloaded from various sources: [LMSys](https://huggingface.co/datasets/lmsys/lmsys-chat-1m), 
[ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered), and 
[SWEBench](https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20240820_honeycomb).

```bash
cd marconi
# install gdown to download the file
# NOTE(ruipan): gdown is already included in the conda environments. The following only needs to be done
# in case the gdown command runs into a "Permission Denied" issue.
# See https://github.com/wkentaro/gdown/issues/43#issuecomment-621356443 for more details.
pip install -U --no-cache-dir gdown --pre

# download the tar file
gdown --fuzzy 'https://drive.google.com/file/d/1D8f68sBWJHyCfJZdEYCBK2M0iHmSDE6M/view?usp=sharing'

# uncompress the tar file
tar -xzvf traces.tar.gz && rm traces.tar.gz

# create the directory for storing output logs and pickles files that record experiment stats
mkdir logs && mkdir results && mkdir -p figures/eval
```

Once all the dependencies has been set up, the directory should have the following structure:

```
--marconi (this repo)
  --data (cloned with the repo)
  --figures (empty, created in the previous step, and will be populated by running plotting scripts)
  --logs (empty, created in the previous step, and will be populated by running the experiments)
  --plotting (cloned with the repo)
  --results (empty, created in the previous step, and will be populated by running the experiments)
  --traces (downloaded and decompressed in the previous step)
  --utils (cloned with the repo)
```

## Experiment workflow

> :warning: Due to the time required to run all experiments, we recommend executing the following instructions in `tmux` or `screen`. 
On our 32-core CPU, a single {LMSys, ShareGPT, SWEBench} trace takes an average of about {30s, 5s, 5-10mins} to finish.

> To help with understanding Marconi's key components, we have prepared a script that includes toy examples of creating and populating a radix tree in `toy_example.py`.

The sweep of all experiments (combinations of different cache sizes and request arrival patterns on different datasets) can be done by running `bash run_all_experiments.sh`, which invokes `policy_exploration.py` and does the following for each experiment configuration (dataset/arrival rate/cache size combination):

- Runs vLLM+
- Runs eviction policy V1, which represents SGLang+
- Runs eviction policy V2, which represents Marconi (was v9 in mlsys experiments)
- Runs eviction policy V3, which represents an offline-optimal, static-α oracle policy (the results weren't included in the paper). This policy sweeps over possible values of α and selects the one that maximizes the hit rate (was v4 in mlsys experiments)

Running `bash run_all_experiments.sh` creates three log files in `/logs`: `lmsys.txt`, `sharegpt.txt`, and `swebench.txt`. Each file contains the output log of all evaluations on the sweep of configurations for this dataset. Once the log files have been generated, plotting scripts can be run to analyze and plot the results.

## Evaluation and expected result

All plotting scripts are under `/plotting` and can be run once the sweep of experiments on all configurations has finished:
- Fig. 7 can be reproduced by running `token_hit_rate.py`
- Fig. 8 can be reproduced by running `sglang_comparison.py`

The two scripts above plot the main results. The following scripts either do not exactly reproduce the figures in the paper (due to some experiments being run on earlier versions of the codebase) or contain hardcoded numbers (handpicked from log files). As a result, we have opted to exclude them from the artifact evaluation process. However, these scripts are still runnable!

- Fig. 10 can be reproduced by running `fine_grained_analysis.py`
- Fig. 11 can be reproduced by running `microbenchmark_contention.py`
- Fig. 12a can be reproduced by running `microbenchmark_layer_composition.py`
- Fig. 12b can be reproduced by running `microbenchmark_dstate.py`
- Fig. 13a and 13b can be reproduced by running `microbenchmark_arrivalrate.py`

## Experiment customization

- Additional eviction policies can be easily implemented in `radix_cache_hybrid.py` by adding a new `evict_policy_version`
- Additional model configurations can be applied by editing the default configurations (NVIDIA's Attention-Mamba2 7B Hybrid model) in `policy_exploration.py`
- Additional datasets can be evaluated by producing compatible traces as the ones generated by `/utils/generate_trace.py`

## Methodology

Submission, reviewing, and badging methodology:

- http://cTuning.org/ae/submission-20190109.html
- http://cTuning.org/ae/reviewing-20190109.html
- https://www.acm.org/publications/policies/artifact-review-badging
