# MMO-SBSE
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)


## Overview
This repository contains the implementation and experimental code for the paper *Specializing Multi-Objectivization as A Simplistic Baseline for Software Engineering Optimization*.


## Introduction
In Search-Based Software Engineering (SBSE), multi-objective optimization has been increasingly adopted to mitigate the risk of single-objective methods being trapped in local optima and to identify better solutions. Prior work on **Meta Multi-Objectivization** in software configuration tuning has demonstrated that combining target and auxiliary objectives can further enhance the effectiveness of multi-objective techniques. Building on this work, we explore the broader applicability of Meta Multi-Objectivization in SBSE, leveraging the fact that it operates at the model level and can thus be generalized beyond configuration tuning. In this study, we propose six new forms of auxiliary objectives and conduct an extensive empirical study across 10 SBSE problems, comprising a total of 224 instances. Our results show that Meta Multi-Objectivization equipped with the most effective auxiliary objectives consistently outperforms conventional MMO approaches that rely solely on the original auxiliary objectives. To reduce the cost of identifying the optimal auxiliary objectives for each instance, we further introduce \approach, a predictive tool that automatically estimates the best auxiliary objective form for a given SBSE problem instance. Evaluation shows that \approach\ outperforms random selection in 70.5%–80.4% of the instances. This study not only provides practical guidance for selecting effective auxiliary objectives in SBSE, but also offers a foundation for future research on predictive support for model-level multi-objective optimization.


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
#### 3.1 Run Multi-Objectivization (NAS as Reference)
> Note: Prepare NAS datasets first (see [_EvoXBench_](https://github.com/EMI-Group/evoxbench)).

```bash
# Navigate to NAS folder
cd Code/NAS

# Run all modes with 30 CPU cores, seeds 0-4 (parallel enabled)
python mmo_nas.py --cpu-cores 50 --seeds 0-4 --mode all

# Run only 'ft_fa' mode with parallel disabled, single seed 5
python mmo_nas.py --no-parallel --mode ft_fa --seeds 5

# Run 'gaussian_fa' mode with seeds 1,3,5 (CSV format)
python mmo_nas.py --mode gaussian_fa --seeds 1,3,5
```

#### 3.2 Run Prediction Pipeline (NAS as Reference)
```bash
# Navigate to NAS Feature folder
cd Code/NAS/Feature

# Run default feature processing pipeline(specific parameters can be set in the code)
python feature_process.py
```

#### 5.3 Run Mode Prediction
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
