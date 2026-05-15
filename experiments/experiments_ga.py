import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import concurrent.futures
import re

# --- FIX: Force Python to recognize the root directory ---
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from src.common.parser import Parser
from src.common.problem import CVRPProblem 
from src.ga.multimodal_genetic_algorithm import PassiveArchiveGA, SequentialNichingGA

def get_route_edges(route):
    edges = set()
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        edges.add((min(u, v), max(u, v))) 
    return edges

def calculate_jaccard(route_a, route_b):
    edges_a = get_route_edges(route_a)
    edges_b = get_route_edges(route_b)
    if not edges_a and not edges_b: return 0.0
    intersection = len(edges_a.intersection(edges_b))
    union = len(edges_a.union(edges_b))
    return 1.0 - (intersection / union)

def calculate_niche_metrics(top3_routes):
    if not top3_routes or len(top3_routes) < 2:
        return np.nan, np.nan, np.nan
    costs = [niche[1] for niche in top3_routes]
    mean_fitness = np.mean(costs)
    std_fitness = np.std(costs)
    
    jaccard_scores = []
    routes_only = [niche[0] for niche in top3_routes]
    for r1, r2 in itertools.combinations(routes_only, 2):
        jaccard_scores.append(calculate_jaccard(r1, r2))
        
    mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
    return mean_fitness, std_fitness, mean_jaccard

# =====================================================================
# THE WORKER FUNCTION (Runs independently on separate CPU cores)
# =====================================================================
def _worker_run(run_id, instance_file_path, instance_name, pop_size, gen_archive, gen_seq_niche, sim_threshold, ls_prob):
    # 1. Parse inside the worker to avoid memory/pickling issues between CPU cores
    parser = Parser(str(instance_file_path))
    nodes, demands, capacity = parser.parse()
    
    match = re.search(r'-k(\d+)', instance_name)
    truck_limit = int(match.group(1)) if match else 25
    cvrp_instance = CVRPProblem(nodes, demands, capacity, truck_limit) 

    # --- EXPERIMENT A: Archive Method ---
    start_time_a = time.perf_counter()
    ga_archive = PassiveArchiveGA(cvrp_instance, pop_size=pop_size)
    top3_archive, _ = ga_archive.run(
        generations=gen_archive, 
        mutation_rate=0.20, 
        similarity_threshold=sim_threshold,
        local_search_prob=ls_prob
    )
    exec_time_archive = time.perf_counter() - start_time_a
    mean_fit_a, std_fit_a, mean_jac_a = calculate_niche_metrics(top3_archive)
    
    archive_result = {
        'run': run_id,
        'mean_fitness': mean_fit_a,
        'std_fitness': std_fit_a,
        'mean_jaccard_distance': mean_jac_a,
        'n_evaluations': pop_size * gen_archive, 
        'n_generations': gen_archive,
        'experiment_id': instance_name,
        'tiempo': exec_time_archive
    }

    # --- EXPERIMENT B: Sequential Tabu ---
    start_time_s = time.perf_counter()
    ga_seq = SequentialNichingGA(cvrp_instance, pop_size=pop_size)
    top3_seq, _ = ga_seq.run_sequential(
        num_niches=3,
        generations_per_niche=gen_seq_niche,
        mutation_rate=0.05, 
        similarity_threshold=sim_threshold,
        local_search_prob=ls_prob
    )
    exec_time_seq = time.perf_counter() - start_time_s
    mean_fit_s, std_fit_s, mean_jac_s = calculate_niche_metrics(top3_seq)
    
    sequential_result = {
        'run': run_id,
        'mean_fitness': mean_fit_s,
        'std_fitness': std_fit_s,
        'mean_jaccard_distance': mean_jac_s,
        'n_evaluations': pop_size * (gen_seq_niche * 3),
        'n_generations': gen_seq_niche * 3,
        'experiment_id': instance_name,
        'tiempo': exec_time_seq
    }

    return archive_result, sequential_result


def run_batch_experiments():
    instances_dir = Path("data") 
    base_out_dir = Path("experiments") / "ga"
    
    out_archive = base_out_dir / "feasibility_archive"
    out_sequential = base_out_dir / "feasibility_sequential"
    out_archive.mkdir(parents=True, exist_ok=True)
    out_sequential.mkdir(parents=True, exist_ok=True)

    # 35 Runs required for Mann-Whitney U Test validity
    N_RUNS = 35  
    SIMILARITY_THRESHOLD = 0.20
    LOCAL_SEARCH_PROB = 0.80

    instance_files = list(instances_dir.glob("*.vrp"))
    # Sort files by size (smallest first) so you get CSVs generating quickly at the start
    instance_files.sort(key=lambda x: int(re.search(r'-n(\d+)', x.stem).group(1)) if re.search(r'-n(\d+)', x.stem) else 0)
    
    print(f"Found {len(instance_files)} instances. Firing up CPU cores...\n")

    # Leave 1 or 2 cores free so your computer doesn't completely freeze
    max_cores = max(1, os.cpu_count() - 2) 

    for instance_file in instance_files:
        instance_name = instance_file.stem
        
        # --- ADAPTIVE PARAMETER SCALING (The Secret Sauce for Speed) ---
        nodes_match = re.search(r'-n(\d+)', instance_name)
        num_nodes = int(nodes_match.group(1)) if nodes_match else 50
        
        if num_nodes <= 100:
            pop_size, gen_archive, gen_seq_niche = 100, 200, 80
        elif num_nodes <= 200:
            pop_size, gen_archive, gen_seq_niche = 60, 100, 40
        else: # Huge instances (200+ nodes)
            pop_size, gen_archive, gen_seq_niche = 40, 60, 25

        print(f"=== Processing: {instance_name} ({num_nodes} nodes) | Pop: {pop_size} | Cores: {max_cores} ===")
        
        archive_log = []
        sequential_log = []

        # --- MULTIPROCESSING: Run all 35 seeds simultaneously ---
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
            # Submit all 35 runs to the CPU pool
            futures = {
                executor.submit(
                    _worker_run, run_id, instance_file, instance_name, 
                    pop_size, gen_archive, gen_seq_niche, SIMILARITY_THRESHOLD, LOCAL_SEARCH_PROB
                ): run_id for run_id in range(N_RUNS)
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                print(f"  Progress: {completed}/{N_RUNS} runs completed...", end="\r")
                try:
                    res_arch, res_seq = future.result()
                    archive_log.append(res_arch)
                    sequential_log.append(res_seq)
                except Exception as e:
                    print(f"\n  [!] Error on run: {e}")

        print(f"\n  Finished {instance_name}. Saving CSVs...\n")

        pd.DataFrame(archive_log).to_csv(out_archive / f"experiment_{instance_name}.csv", index=False)
        pd.DataFrame(sequential_log).to_csv(out_sequential / f"experiment_{instance_name}.csv", index=False)

if __name__ == "__main__":
    run_batch_experiments()