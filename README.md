# MMO-SBSE
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)


## Overview
This repository contains the implementation and experimental code for the paper *Specializing Multi-Objectivization as A Simplistic Baseline for Software Engineering Optimization*.


## Introduction
Given the discrete nature of many software engineering (SE) optimization problems, trapping at local optima remains a crucial issue in the optimizer designs. Meta Multi-Objectivization  (MMO) is an optimization model for overcoming local optima by introducing an extra, auxiliary objective. Yet, MMO is originally designed only for configuration tuning, leaving its suitability for other SE problems in doubt, particularly regarding the choice of the auxiliary objective. In this paper, we study the possibility and resolution for specializing MMO as a general baseline across a range of SE optimization problems, providing a fundamental counterpart for future SE optimizers that seek to mitigate the issue of local optima. We do so by providing both theoretical remit and empirical evidence: we analytically show how the choice of the auxiliary objective is crucial while demonstrating the effectiveness of MMO under 10 SE optimization problems and 224 cases when the right one can be chosen, i.e., MMO is able to outperform the other state-of-the-art and general-purpose optimizers in 77\% and 92\% of the cases, respectively. Drawing on the theoretical understanding and empirical findings, we then propose L2RAC, a tool that automatically recommends the best auxiliary objective choice for a given SE problem/case, hence helping to specialize MMO therein. Through extensive experiments, we reveal that, in general, L2RAC~can more accurately rank the choice of auxiliary objective for MMO under a case than random guessing by up to 60\%, saving at least 97\% of the evaluation cost that would otherwise be required to conduct trial-and-error when using MMO.

## SBSE Problem Abbreviations
The following table maps folder abbreviations (in `Code/`) to corresponding SBSE problems:

| Abbreviation | SBSE Problem                  |
|--------------|-------------------------------|
| NAS          | Neural Architecture Search    |
| NRP          | Next Release Problem          |
| SCT          | Software Configuration Tuning |
| SEE          | Software Effort Estimation    |
| SDP          | Software Defect Prediction    |
| SPLT         | Software Product Lines Testing|
| SPSP         | Software Project Scheduling   |
| TPLM         | Third Party Library Migration |
| WS           | Workflow Scheduling           |
| WSC          | Web Service Composition       |


## Key Features
- **Generalized multi-objectivization framework**: A unified framework exploring six domain-independent auxiliary objective generation strategies.
- **Large-scale cross-domain benchmark**: The first systematic evaluation covering 10 heterogeneous SBSE tasks (224 instances) for multi-objectivization generality.
- **Systematic performance validation**: Evaluated across three dimensions, outperforming original MMO (96.9% instances), domain SOTA (86.2%), and mainstream SBSE optimizers (95.1%).
- **Cross-domain optimizer comparison**: First cross-task comparison of three general-purpose SBSE optimizers, filling gaps in comparative SBSE research.
- **Predictive auxiliary-objective selector**: A feature-based model recommending optimal auxiliary objectives for SBSE instances, avoiding costly trial-and-error.


## Repository Structure
All folders under the Code directory (e.g., NAS, NRP, SCT) share the same structure; only the NAS-related code structure is expanded here:

```
MMO-SBSE/
├── README.md
├── requirements.txt
├── Code/
│   ├── NAS/
│   │   ├── README.md
│   │   ├── Datasets/  # Download and set up via the EvoXBench
│   │   │   ├── data/
│   │   │   └── database/
│   │   ├── Feature/
│   │   │   ├── feature_process.py  # Concatenate features and the corresponding mode ranking information
│   │   │   ├── multi_feature.py  # Sample data and calculate multi-objective space features
│   │   │   ├── single_feature.py  # Calculate single-objective space features
│   │   │   └── utils/
│   │   │       ├── calculate_rank.py  # Calculate mode rank
│   │   │       └── multi_feature_compute.py  # Multi-objective space feature calculation function
│   ├── NRP/  # Other SBSE problem folders (structure same as NAS, not expanded)
│   ├── SCT/
│   ├── SEE/
│   ├── SDP/
│   ├── SPLT/
│   ├── SPSP/
│   ├── TPLM/
│   ├── WS/
│   ├── WSC/
│   ├── Utils/
│   │   └── remove_duplicates.py  # Remove duplicates in the process of mmo
│   ├── mode_predict.py  # Predict the ranking of each mode
│   └── random_predict.py  # Baseline function for randomly guessing rankings
└── RQS/  # Supplementary Results for Each RQ in the Paper
```

## Quick Start
### Prerequisites
- Python 3.9


### Installation
1. Clone the repository:
```bash
git clone https://github.com/ideas-labo/mmo-sbse.git
cd mmo-sbse
```

2. Create and activate an Anaconda environment:
```bash
conda create -n mmo-sbse python=3.9 -y
conda activate mmo-sbse
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Basic Usage
### 1. Core Configuration Parameters (mmo_*.py)
The `mmo_*.py` scripts (e.g., `mmo_nas.py`) support global configuration parameters (adjustable in the script or via command-line arguments):

| Parameter          | Default Value          | Description                                                                 |
|--------------------|------------------------|-----------------------------------------------------------------------------|
| `USE_PARALLEL`     | `True`                 | Enable/disable parallel processing (set to `False` to disable)              |
| `CPU_CORES`        | `50`                   | Number of CPU cores used for parallel execution                            |
| `MAX_RUNTIME`      | `24 * 3600` (24h)      | Maximum runtime for experiments (in seconds)                                |
| `MODES`            | See Mode Parameters    | List of auxiliary objective modes to run                                    |
| `SEEDS`            | `range(0, 10)`         | Random seeds for reproducibility                                            |

### 2. Mode Parameters for Auxiliary Objectives
The `MODES` argument maps to different auxiliary objective strategies in the paper:

| Mode Name                | Corresponding Concept in Paper | Description                          |
|--------------------------|---------------------------------|--------------------------------------|
| `ft_fa`                  | Plain Multi-Objectivization     | Plain multi-objectivization strategy |
| `g1_g2`                  | Original Auxiliary Objectives   | Original auxiliary objective design  |
| `penalty_fa`             | Penalty Auxiliary Objectives    | Auxiliary objectives with penalty    |
| `gaussian_fa`            | Gaussian Noise Objectives       | Auxiliary objectives with Gaussian noise |
| `reciprocal_fa`          | Reciprocal Auxiliary Objectives | Reciprocal-based auxiliary objectives |
| `age_maximization_fa`    | Age Auxiliary Objectives        | Auxiliary objectives based on age maximization |
| `novelty_maximization_fa`| Novelty Auxiliary Objectives    | Auxiliary objectives based on novelty maximization |
| `diversity_fa`           | Grid Diversity Auxiliary Objectives | Auxiliary objectives based on grid diversity |


### 3. Example Execution
#### 3.1 Run MMO (NAS as Reference)
> Note: Prepare NAS datasets first (see [_EvoXBench_](https://github.com/EMI-Group/evoxbench)).

```bash
# Navigate to NAS folder
cd Code/NAS

# Run all modes with 50 CPU cores, seeds 0-4 (parallel enabled)
python mmo_nas.py --cpu-cores 50 --seeds 0-4 --mode all

# Run only 'ft_fa' mode with parallel disabled, single seed 5
python mmo_nas.py --no-parallel --mode ft_fa --seeds 5

# Run 'gaussian_fa' mode with seeds 1,3,5 (CSV format)
python mmo_nas.py --mode gaussian_fa --seeds 1,3,5
```

#### 3.2 Compute Features for Prediction
We use the [_ScottKnott Effect Size Difference (ESD) test_](https://github.com/klainfo/ScottKnottESD) (Version 3.0, development branch) for ranking modes.
```bash
# 1. Navigate to NAS Feature folder
cd Code/NAS/Feature

# 2. Run calculate_rank.py to generate ranking information for modes
python utils/calculate_rank.py

# 3. Run feature processing pipeline (specific parameters can be set in the code)
python feature_process.py
```

#### 3.3 Run Mode Prediction
```bash
# Navigate to Code root directory
cd Code

# Predict auxiliary objective rankings
python mode_predict.py

# Run random selection baseline for comparison
python random_predict.py
```

## Reproducing Paper Results
### 1. RQ Results
Run the `mmo_*.py` script in the target SBSE problem folder to obtain multi-objectivization results for the three research questions (RQs) in the paper.

```bash
# Example: Reproduce RQ results for NAS (all modes, default config)
cd Code/NAS
python mmo_nas.py
```

### 2. Prediction Results
1. Generate multi-objective/single-objective features for a target SBSE problem (NAS as example):
```bash
cd Code/NAS/Feature
python feature_process.py
```

2. Predict auxiliary objective rankings (run from Code root directory):
```bash
cd Code
python mode_predict.py
```

3. (Optional) Run random selection baseline for comparison:
```bash
cd Code
python random_predict.py
```


## Comparative Algorithms' Literature
### SOTA Algorithms
- NAS: [Construction of hierarchical neural architecture search spaces based on context-free grammars](https://proceedings.neurips.cc/paper_files/paper/2023/file/4869f3f967dfe954439408dd92c50ee1-Paper-Conference.pdf)
- NRP: [Solving the Large Scale Next Release Problem with a Backbone-Based Multilevel Algorithm](https://doi.org/10.1109/TSE.2011.92)
- SCT: [PromiseTune: Unveiling Causally Promising and Explainable Configuration Tuning](https://doi.org/10.48550/arXiv.2507.05995)
- SEE: [Multi-Objective Software Effort Estimation: A Replication Study](https://doi.org/10.1109/TSE.2021.3083360)
- SDP: [Classification framework for faulty-software using enhanced exploratory whale optimizer-based feature selection scheme and random forest ensemble learning](https://doi.org/10.1007/s10489-022-04427-x)
- SPLT: [Solving the t-Wise Coverage Maximum Problem via Effective and Efficient Local Search-Based Sampling](https://doi.org/10.1145/3688836)
- TPLM: [Search-Based Third-Party Library Migration at the Method-Level](https://doi.org/10.1007/978-3-031-02462-7\_12)
- WS: [A hybrid genetic algorithm for optimization of scheduling workflow applications in heterogeneous computing systems](https://doi.org/10.1016/j.jpdc.2015.10.001)
- WSC: [A Hybrid Strategy Improved Whale Optimization Algorithm for Web Service Composition](https://doi.org/10.1093/comjnl/bxab187)
### General-Purpose SBSE Optimizers
- SWAY: [“Sampling” as a baseline optimizer for search-based software engineering](https://ieeexplore.ieee.org/document/8249828/)
- LINE and LITE: [BINGO! Simple Optimizers Win Big if Problems Collapse to a Few Buckets](https://ieeexplore.ieee.org/document/7352396)


## Data Availability

The raw experimental results and generated datasets supporting the findings of this study are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18200015.svg)](https://doi.org/10.5281/zenodo.18200015)

The dataset includes the following components organized by research questions:

### 1. RQ Experimental Results
- **Complete experimental results for all 10 SBSE problems (224 instances)**:
  - PMO (Plain Multi-Objectivization) results
  - MMO results with original auxiliary objectives
  - MMO results with six different auxiliary objective strategies (as defined in Mode Parameters)
  - Comparative results with domain-specific SOTA algorithms
  - Baseline results from general-purpose SBSE optimizers (LINE, LITE, SWAY)
  - Random Search (RS) benchmark results

### 2. Predictive Modeling Data
- **rANK data** for modes
- **Feature data** used for training the auxiliary objective prediction model
- **Sampling data** used to compute the feature values
- **Prediction results** of auxiliary objective rankings across all SBSE problem instances


