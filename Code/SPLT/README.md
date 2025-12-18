# SPLT Module for MMO-SBSE
This module contains the Meta Multi-Objectivization (MMO) extension code for the **Software Product Lines Testing (SPLT)** task in the MMO-SBSE project, built on top of the open-source implementation of *Automated Test Suite Generation for Software Product Lines Based on Quality-Diversity Optimization*.


## Environment Requirement
- **Java Development Kit (JDK)**: JDK 11 (required to compile and run our provided Java functions and JAR packages)


## Overview
The SPLT module is built on the open-source project [SPLTestingMAP](https://github.com/gzhuxiangyi/SPLTestingMAP) (paper: *Automated Test Suite Generation for Software Product Lines Based on Quality-Diversity Optimization*). We **do not redistribute the full code of SPLTestingMAP**—only our newly developed MMO-related Java functions. To use this module, follow the directory mapping rules below to integrate the original project with our extension code.


## Directory Integration Rules
### Step 1: Rename & Place the Original Project
1. Clone the original `SPLTestingMAP` repository from GitHub:  
   `git clone https://github.com/gzhuxiangyi/SPLTestingMAP.git`
2. Rename the entire `SPLTestingMAP` folder to `SPLT`.
3. Move the renamed `SPLT` folder into the `Code/` directory of our mmo-sbse project (final path: `MMO-SBSE/Code/SPLT/`).

### Step 2: Integrate Our MMO Extension Code
Our newly developed MMO-related code is packaged in an `mmo` folder—copy this folder into the `src/` directory of the renamed `SPLT` project (consistent with the `src` directory structure of the original SPLTestingMAP):


## Our Provided MMO Extension Code (Java Functions)
We only provide newly developed MMO-related tool functions (with default parameters for out-of-the-box use), no duplication of the original SPLTestingMAP code:
1. **GenerateFA.java**  
   Tool function for auxiliary objective generation (core logic of MMO), implementing 6 domain-independent auxiliary objective strategies (penalty, Gaussian noise, reciprocal, age maximization, novelty maximization, grid diversity) tailored for SPLT scenarios.
2. **NSGA2Optimizer.java**  
   Core MMO optimization function, extending the original NSGA-II algorithm to support MMO auxiliary objectives (default parameters for population size, iteration count, etc. are pre-configured).
3. **ProductSampler.java**  
   Sampling function for feature calculation, generating product samples required for MMO feature analysis (default sampling size and method are pre-set).

> All above Java functions are included in the `mmo` folder, along with compiled JAR packages (e.g., `NSGA2Optimizer.jar`, `SATSolvingBasedTesting.jar`) for direct invocation in the original SPLT workflow.

