import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
import itertools

# =====================================================================
# IMPORTS MATCHING YOUR WORKSPACE ARCHITECTURE
# =====================================================================
from src.common.problem import load_cvrp_instance 
from src.ga.multimodal_genetic_algorithm import PassiveArchiveGA, SequentialNichingGA

def get_route_edges(route):
    """Converts a sequence of nodes into a set of undirected edges."""
    edges = set()
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        edges.add((min(u, v), max(u, v))) # Undirected
    return edges

def calculate_jaccard(route_a, route_b):
    """Calculates the Jaccard distance between two routes."""
    edges_a = get_route_edges(route_a)
    edges_b = get_route_edges(route_b)
    
    if not edges_a and not edges_b:
        return 0.0
        
    intersection = len(edges_a.intersection(edges_b))
    union = len(edges_a.union(edges_b))
    return 1.0 - (intersection / union)

def calculate_niche_metrics(top3_routes):
    """Calculates mean fitness, std deviation, and mean Jaccard of the niches."""
    if not top3_routes or len(top3_routes) < 2:
        return np.nan, np.nan, np.nan
    
    # 1. Fitness Metrics (top3_routes is a list of tuples: [(route, cost), ...])
    costs = [niche[1] for niche in top3_routes]
    mean_fitness = np.mean(costs)
    std_fitness = np.std(costs)
    
    # 2. Jaccard Metrics
    jaccard_scores = []
    routes_only = [niche[0] for niche in top3_routes]
    
    # Compare all unique pairs (e.g., Niche 1v2, 1v3, 2v3)
    for r1, r2 in itertools.combinations(routes_only, 2):
        jaccard_scores.append(calculate_jaccard(r1, r2))
        
    mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
    
    return mean_fitness, std_fitness, mean_jaccard

def run_batch_experiments():
    # --- 1. SETUP PATHS ---
    instances_dir = Path("data") # Verify this points to your .vrp files
    base_out_dir = Path("experiments") / "ga"
    
    out_archive = base_out_dir / "feasibility_archive"
    out_sequential = base_out_dir / "feasibility_sequential"
    
    out_archive.mkdir(parents=True, exist_ok=True)
    out_sequential.mkdir(parents=True, exist_ok=True)

    # --- 2. TUNED HYPERPARAMETERS ---
    N_RUNS = 35  # Number of executions