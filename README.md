## CVRP GROUP 9

CHEN BAO WUE LIANG
IRALDE DOMINGUEZ RAFAEL
MORA MEDINA MANUEL ADRIAN
MORERA HERNANDEZ MIGUEL ANGEL
RIVERA CUEVA JONNY DAVID

# Advanced Metaheuristics for the Capacitated Vehicle Routing Problem (CVRP)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## Project Overview
This repository contains the implementation, tuning, and statistical benchmarking of three distinct nature-inspired metaheuristics applied to the **Capacitated Vehicle Routing Problem (CVRP)**. 

Moving beyond traditional global optimization, this project places a strong emphasis on **Multimodal Optimization**. The goal is not only to minimize the total routing distance but to provide logistics planners with a *portfolio of structurally diverse, high-quality alternative routes* to increase operational resilience against real-world disruptions.

---

## Implemented Metaheuristics

We engineered three distinct algorithmic architectures to tackle the CVRP constraint space, adapting continuous frameworks into discrete combinatorial solvers:

### 1. Genetic Algorithm (GA) - *Multimodal & Memetic*
Our flagship algorithm designed to map the entire search landscape and extract diverse optima.
* **Representation:** Integer-based permutation chromosomes decoded via a greedy splitting procedure.
* **Memetic Refinement:** Integrates a Lamarckian **2-opt local search** to dynamically untangle spatial route intersections within a single generation.
* **Niche Management:** Implements active diversity preservation using **Deterministic Crowding** (parallel exploration) and **Sequential Niching** (Tabu-style memory erasure via fitness penalties).

### 2. Ant Colony Optimization (ACO) - *Probabilistic Constructive*
A swarm-intelligence approach leveraging collective memory.
* **Pheromone Memory:** Uses a continuously updated pheromone matrix to reinforce highly efficient node-to-node transitions.
* **Heuristic Visibility:** Movement is guided probabilistically by inverse distance weighting.
* **Dynamic Feasibility:** The feasible neighborhood is strictly filtered during construction; ants are forced to return to the depot when remaining truck capacity is exhausted.

### 3. Particle Swarm Optimization (PSO) - *Discrete Adaptation*
A highly adapted swarm strategy overcoming dimensionality issues in discrete spaces.
* **Representation:** Particles define position as a permutation of customers ("Giant Tour").
* **Probabilistic Movement:** The traditional continuous "velocity" is redefined as a sequence of transposition operators (swaps). 
* **Swarm Logic:** Swap sequences are calculated probabilistically to guide particles toward their personal best (pBest) and the global best (gBest).

---

## Repository Architecture

The project is structured to separate core algorithmic logic from experimental execution and statistical analysis.

```text
📦 CVRP-Metaheuristics
 ┣ 📂 data/                   # Standardized CVRP benchmark instances (e.g., X-n106-k14)
 ┣ 📂 experiments/            # CSV outputs from batch executions (35-run statistical rigor)
 ┣ 📂 src/                    # Core Python modules (Algorithm classes, parsers, diversity metrics)
 ┣ 📜 run_ga.ipynb            # Execution, tuning, and visualization of the Genetic Algorithm
 ┣ 📜 run_aco.ipynb           # Execution, tuning, and visualization of Ant Colony Optimization
 ┣ 📜 run_pso.ipynb           # Execution, tuning, and visualization of Particle Swarm Optimization
 ┣ 📜 compare_results.ipynb   # Final statistical analysis (Kruskal-Wallis, Friedman) & cross-algorithm benchmarking
 ┣ 📜 setup.py                # Package configuration
 ┣ 📜 requirements.txt        # Python dependencies
 ┗ 📜 README.md               # Project documentation
