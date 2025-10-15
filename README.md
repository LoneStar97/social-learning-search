# Social learning moderates the tradeoffs between efficiency, stability, and equity in group foraging
This repository contains Python scripts and simulation data for studying **collective foraging behaviors** in environments containing both **positive** (resource) and **negative** (risk) patches.  The model explores how **social learning range** ($\rho$) shapes foraging efficiency, behavioral composition, and equity among agents.

---

## 📑 Table of Contents
- [📝 Model Overview](#-model-overview)
- [📂 Model Components](#-model-components)
- [🔬 Simulation Scenarios](#-simulation-scenarios)
	- [Parameters](#parameters)
	- [Base Model](#base-model)
	- [Introducing Negative Patches](#introducing-negative-patches)
- [🚀 Running the Model & Analysis Approach](#-running-the-model--analysis-approach)
- [⚙️ Dependencies](#️-dependencies)

---

## 📝 Model Overview

The model simulates a population of **foraging agents** that navigate a 2D periodic environment populated by **positive targets** (resources) and optionally **negative targets** (hazards).  
Each agent alternates between three behavioral modes:

1. **Exploration ($\mu$ = 1.1 Lévy walk)** — broad search for unknown areas.  
2. **Exploitation ( $\mu$ = 3.0 Lévy walk)** — intensive search near detected resources (**Area-Restricted Search**).  
3. **Targeted walk** — directed motion toward detected targets or social cues.

Agents exchange information over a **social learning range (ρ)**, which governs the spatial scale of communication.  
The model explores how increasing ρ changes:

- **Efficiency ($\eta$)** — group-level foraging success.  
- **Behavioral allocation** — the proportion of time spent exploring, exploiting, or targeted walk. 
- **Stability** — the consistency of resource intake across the group of agents.
- **Equity ($\sigma$)** — fairness of target distribution among agents.  

This model extends previous foraging frameworks by introducing **negative patches**, representing harmful or deceptive areas that reduce the total payoff and test agents’ collective robustness.

---

## 📂 Model Components

The repository consists of three core modules:

### 🧩 1. Base Model (`/BaseModel`)
| File                        | Description                                                                                                                        |
| :-------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| `ModMainFix.py`             | Core simulation engine implementing agent motion, target detection, and interaction logic. Produces case-level `.h5` output files. |
| `plotfunc.py`               | High-level orchestration for time evolution (`timedata`) and visualization (`draw_video`).                                         |
| `periodic_kdtree.py`        | Custom KD-tree structure with **periodic boundary conditions** for efficient neighbor search (adapted from SciPy).                 |
| `Parallel_FixTar_Rv0.01.py` | Parallelized runner for large-scale parameter sweeps (different $\rho$ values). Uses Python `multiprocessing.Pool`.                |
|                             |                                                                                                                                    |

---

### ☢️ 2. Negative Target Model (`/NegativeTargets`)
| File                        | Description                                                                                                                             |
| :-------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| `ModMainFix_Negative.py`    | Extended simulation module integrating **negative targets**. Defines `main_timeana` and `main_video` functions for mixed environments.  |
| `plotfunc.py`               | Controls both positive and negative target runs. Calls `main_timeana` from `ModMainFix_Negative.py` to generate simulation data.        |
| `Parallel_FixTar_Rv0.01.py` | Batch execution script that runs 2000 cases per $\rho$ value. Loads mixed positive/negative target coordinates from `10P600_10NP60.h5`. |
| `periodic_kdtree.py`        | Same as the BaseModel version — ensures spatial periodicity for all distance computations.                                              |

These scripts together simulate **agents operating in environments that include harmful regions**, stored as negative target coordinates (`ntx`, `nty`).

---

### 🎯 3. Target Distribution Generator (`/Target_distribution`)
| File                 | Description                                                             |
| :------------------- | :---------------------------------------------------------------------- |
| `FixTar_Positive.py` | Generates `.h5` datasets containing only positive (resource) targets.   |
| `FixTar_Negative.py` | Generates `.h5` datasets containing both positive and negative targets. |

Each `.h5` file contains multiple groups (`case_i`), each including:
- `tx`, `ty` → coordinates of positive targets  
- `ntx`, `nty` → coordinates of negative targets (for the negative version)

---

### 📈 4. Figures and Visualization (`/Figures`)
| File         | Description                                                                            |
| :----------- | :------------------------------------------------------------------------------------- |
| `Figure1.py` | Schematic of the model setup — agents, radii ($R$, $\rho$), and example trajectories.  |
| `Figure2.py` | Efficiency ($\eta$) vs. social learning range ($\rho$), and phase transition analysis. |
| `Figure3.py` | Equity analysis — PDF of targets collected per agent.                                  |
| `Figure4.py` | Comparison between base and negative-target environments.                              |

---

## 🔬 Simulation Scenarios

### Parameters
|    Symbol    | Parameter Name                                | Description                                                                                                                                                                                                                                                                                                                |
| :----------: | :-------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **$\beta$**  | Controlling parameter for target distribution | The parameter $\beta$ controls the spatial distribution of targets around the target seeds, such that $\beta \to 1$ results in a nearly uniform distribution, while $\beta \to 3$ produces a distribution with tightly clustered targets. The main text presents the results using a target distribution with $\beta = 2$. |
|  **$\rho$**  | Social learning range                         | The distance over which agents perceive or respond to other agents’ social cues. Governs the spatial scale of information sharing.                                                                                                                                                                                         |
|    **Rv**    | Visual detection radius                       | The maximum distance within which an agent can directly detect targets (resources or hazards). Controls the local sensing ability.                                                                                                                                                                                         |
|    **R**     | Interaction radius                            | Defines the neighborhood size for local social communication (used in Area-Restricted Search).                                                                                                                                                                                                                             |
|  **$\mu$**   | Movement exponent                             | Governs the Lévy-type step-length distribution: smaller $\mu$ → more exploratory (long jumps); larger $\mu$ → more exploitative (short, local moves).                                                                                                                                                                      |
|  **$\eta$**  | Foraging efficiency                           | The total number of collected resources normalized by time; quantifies system performance.                                                                                                                                                                                                                                 |
| **$\sigma$** | Equity measure                                | Standard deviation of collected targets across agents; smaller $\sigma$ indicates fairer distribution.                                                                                                                                                                                                                     |
|    **N**     | Number of agents                              | Total population of foragers in the simulation.                                                                                                                                                                                                                                                                            |
|    **T**     | Simulation time                               | Duration (in time steps) for which the simulation runs.                                                                                                                                                                                                                                                                    |


### Base Model
- Environment contains **only positive targets** (resources).  
- Used to characterize baseline foraging efficiency and behavioral patterns.  
- Outputs stored as `Rv{Rv}_rho={ρ}_A{N}.h5`. 

### Introducing Negative Patches
- Adds **negative targets** that act as hazards, reducing net resource gain.  
- These modify agent decisions and broaden the equity distribution.  
- Implemented through `ModMainFix_Negative.py` and executed via `Parallel_FixTar_Rv0.01.py`.  
- Output stored as `Rv{Rv}rho={ρ}_A{N}.h5`.

---

## 🚀 Running the Model & Analysis Approach

### 1️⃣ Generate Target Distributions
```bash
# Positive-only environment
python Target_distribution/FixTar_Positive.py

# Positive + Negative environment
python Target_distribution/FixTar_Negative.py```
```
###  2️⃣ Run Simulations
```bash
# Run based model 
python BaseModel/Parallel_FixTar_Rv0.01.py

# Run model with negative patches
python Parallel_FixTar_Rv0.01.py
```

### 3️⃣ Run Analysis and Plot Figures
```bash
python Figures/Figure1.py   # schematic diagram
python Figures/Figure2.py   # efficiency and behavioral ratios
python Figures/Figure3.py   # equity PDFs
python Figures/Figure4.py   # negative patch comparisons
```

## ⚙️ Dependencies
```bash
Python >= 3.9
numpy
scipy
matplotlib
h5py
sklearn
tqdm
pandas
ipywidgets
```

