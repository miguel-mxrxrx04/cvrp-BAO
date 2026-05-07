import time
import statistics
from src.ACO.ACO import InspyredACOSolver


def load_vrp_data(filepath):
    """Reads a standard .vrp file and extracts coordinates, demands, and capacity."""
    nodes = {}
    demands = {}
    capacity = 0

    print(f"Reading data file: {filepath}...")

    with open(filepath, 'r') as file:
        lines = file.readlines()

    current_section = None

    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            continue

        if line.startswith("CAPACITY"):
            parts = line.replace(":", " ").split()
            capacity = int(parts[-1])
            continue

        if line.startswith("NODE_COORD_SECTION"):
            current_section = "coords"
            continue
        elif line.startswith("DEMAND_SECTION"):
            current_section = "demands"
            continue
        elif line.startswith("DEPOT_SECTION"):
            current_section = "depot"
            continue

        if current_section == "coords" and line[0].isdigit():
            parts = line.split()
            node_id = int(parts[0])
            nodes[node_id] = (float(parts[1]), float(parts[2]))

        elif current_section == "demands" and line[0].isdigit():
            parts = line.split()
            node_id = int(parts[0])
            demands[node_id] = int(parts[1])

    return nodes, demands, capacity


def run_experiment(config_name, filepath, n_runs=30, **aco_params):
    print(f"\n🚀 Starting Experiment: {config_name}")
    print(f"Running {n_runs} times. Please wait...\n")

    nodes, demands, capacity = load_vrp_data(filepath)

    all_best_costs = []
    all_execution_times = []

    # Track the best overall execution for plotting
    overall_best_cost = float('inf')
    best_solver_instance = None
    best_alternatives = None

    for run in range(1, n_runs + 1):
        start_time = time.time()

        # Pass the run index as the seed to ensure stochastic variance across runs
        solver = InspyredACOSolver(
            nodes=nodes,
            demands=demands,
            capacity=capacity,
            seed=run,
            **aco_params  # Unpack dynamic hyperparameters
        )

        best_route, best_cost, alternatives = solver.solve(n_alternatives=3)

        end_time = time.time()
        execution_time = end_time - start_time

        all_best_costs.append(best_cost)
        all_execution_times.append(execution_time)

        # Update global best result
        if best_cost < overall_best_cost:
            overall_best_cost = best_cost
            best_solver_instance = solver
            best_alternatives = alternatives

        # Print progress periodically to avoid console spam
        if run % 5 == 0 or run == 1:
            print(f"  -> Run {run}/{n_runs} | Cost: {best_cost:.2f} | Time: {execution_time:.2f}s")

    # Statistical calculations
    mean_cost = statistics.mean(all_best_costs)
    stdev_cost = statistics.stdev(all_best_costs) if n_runs > 1 else 0.0
    mean_time = statistics.mean(all_execution_times)
    worst_overall_cost = max(all_best_costs)

    print("\n" + "=" * 50)
    print(f"📊 FINAL RESULTS: {config_name}")
    print("=" * 50)
    print(f"Best Cost (Min)      : {overall_best_cost:.2f}")
    print(f"Worst Cost (Max)     : {worst_overall_cost:.2f}")
    print(f"Average Cost (Mean)  : {mean_cost:.2f} ± {stdev_cost:.2f}")
    print(f"Average Time/Run     : {mean_time:.2f} seconds")
    print("=" * 50)

    # Show plots only for the best overall run
    print("\nGenerating plots for the best execution...")
    best_solver_instance.plot_convergence()
    if best_alternatives:
        best_solver_instance.plot_alternative_comparison(overall_best_cost, best_alternatives)


if __name__ == "__main__":

    test_file = "../data/X-n106-k14.vrp"

    run_experiment(
        config_name="Baseline Configuration",
        filepath=test_file,
        n_runs=1,
        n_ants=40,
        n_iterations=100,
        alpha=1.0,
        beta=3.0,
        evaporation=0.25
    )