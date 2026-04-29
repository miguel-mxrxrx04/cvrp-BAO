import math
import random
from copy import deepcopy
import inspyred


class InspyredACOSolver:
    def __init__(
            self,
            nodes,
            demands,
            capacity,
            depot_id=1,
            n_ants=40,
            n_iterations=200,
            alpha=1.0,
            beta=3.0,
            evaporation=0.25,
            q=100.0,
            seed=42,
    ):
        self.nodes = nodes
        self.demands = demands
        self.capacity = capacity
        self.depot_id = depot_id

        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q

        # PRNG instance required by inspyred
        self.prng = random.Random(seed)

        self.customers = [i for i in nodes if i != depot_id]
        self.dist = self._build_distance_matrix()

        # Initialize pheromone matrix
        self.pheromone = {
            (i, j): 1.0 for i in nodes for j in nodes if i != j
        }

        # History dictionary for plotting
        self.history = {
            "iteration": [],
            "best": [],
            "iteration_best": [],
            "average": [],
        }

        # Archive for multimodal solutions
        self.archive = []

    def _build_distance_matrix(self):
        dist = {}
        for i, (x1, y1) in self.nodes.items():
            for j, (x2, y2) in self.nodes.items():
                if i != j:
                    dist[(i, j)] = math.hypot(x1 - x2, y1 - y2)
        return dist

    def route_distance(self, route):
        if not route:
            return 0.0

        total = self.dist[(self.depot_id, route[0])]
        for a, b in zip(route, route[1:]):
            total += self.dist[(a, b)]
        total += self.dist[(route[-1], self.depot_id)]
        return total

    def solution_distance(self, solution):
        return sum(self.route_distance(route) for route in solution)

    def _choose_next_customer(self, current, feasible):
        scores = []

        for customer in feasible:
            tau = self.pheromone[(current, customer)] ** self.alpha
            eta = (1.0 / (self.dist[(current, customer)] + 1e-9)) ** self.beta
            scores.append(tau * eta)

        total = sum(scores)

        if total == 0:
            return self.prng.choice(feasible)

        r = self.prng.random() * total
        cumulative = 0.0

        for customer, score in zip(feasible, scores):
            cumulative += score
            if cumulative >= r:
                return customer

        return feasible[-1]

    def construct_solution(self):
        unvisited = set(self.customers)
        solution = []

        while unvisited:
            route = []
            load = 0
            current = self.depot_id

            while True:
                feasible = [
                    c for c in unvisited
                    if load + self.demands[c] <= self.capacity
                ]

                if not feasible:
                    break

                next_customer = self._choose_next_customer(current, feasible)

                route.append(next_customer)
                unvisited.remove(next_customer)
                load += self.demands[next_customer]
                current = next_customer

            solution.append(route)

        return solution

    def two_opt_route(self, route):
        best = route[:]
        improved = True

        while improved:
            improved = False

            for i in range(len(best) - 1):
                for j in range(i + 2, len(best)):
                    candidate = best[:i] + best[i:j][::-1] + best[j:]

                    if self.route_distance(candidate) < self.route_distance(best):
                        best = candidate
                        improved = True

        return best

    def improve_solution(self, solution):
        return [self.two_opt_route(route) for route in solution]

    def _evaporate_pheromones(self):
        for edge in self.pheromone:
            self.pheromone[edge] *= (1.0 - self.evaporation)
            self.pheromone[edge] = max(self.pheromone[edge], 1e-6)

    def _deposit_pheromones(self, solution, cost):
        deposit = self.q / cost

        for route in solution:
            full_route = [self.depot_id] + route + [self.depot_id]

            for a, b in zip(full_route, full_route[1:]):
                if a != b:
                    self.pheromone[(a, b)] += deposit
                    self.pheromone[(b, a)] += deposit

    def _edge_set(self, solution):
        edges = set()

        for route in solution:
            full_route = [self.depot_id] + route + [self.depot_id]
            for a, b in zip(full_route, full_route[1:]):
                edges.add(tuple(sorted((a, b))))

        return edges

    def diversity(self, sol_a, sol_b):
        edges_a = self._edge_set(sol_a)
        edges_b = self._edge_set(sol_b)

        if not edges_a and not edges_b:
            return 0.0

        overlap = len(edges_a & edges_b)
        union = len(edges_a | edges_b)

        return 1.0 - overlap / union

    def is_valid(self, solution):
        seen = []

        for route in solution:
            load = sum(self.demands[c] for c in route)

            if load > self.capacity:
                return False

            seen.extend(route)

        return sorted(seen) == sorted(self.customers)

    # =====================================================================
    # INSPYRED FRAMEWORK COMPONENTS
    # =====================================================================

    def _generate_ants(self):
        """Helper method to construct and improve an entire ant population."""
        solutions = []
        for _ in range(self.n_ants):
            sol = self.construct_solution()
            sol = self.improve_solution(sol)
            solutions.append(sol)
        return solutions

    def _aco_generator(self, random, args):
        """[INSPYRED] Generates the initial population for the first generation."""
        return self._generate_ants()

    def _aco_variator(self, random, candidates, args):
        """[INSPYRED] Generates brand new ants for subsequent generations."""
        return self._generate_ants()

    def _aco_evaluator(self, candidates, args):
        """[INSPYRED] Calculates fitness (distance) for each ant. Penalizes invalid routes."""
        fitnesses = []
        for sol in candidates:
            if self.is_valid(sol):
                fitnesses.append(self.solution_distance(sol))
            else:
                fitnesses.append(float('inf'))  # Heavy penalty for invalid solutions
        return fitnesses

    def _aco_replacer(self, random, population, parents, survivors, args):
        """[INSPYRED] Handles pheromone updates and replaces the old population."""
        # 1. Evaporate pheromones globally
        self._evaporate_pheromones()

        # 2. Deposit pheromones based on the best ant of the current iteration (survivors)
        valid_survivors = [ind for ind in survivors if ind.fitness != float('inf')]
        if valid_survivors:
            iteration_best = min(valid_survivors, key=lambda x: x.fitness)
            self._deposit_pheromones(iteration_best.candidate, iteration_best.fitness)

        # 3. Generational replacement: new ants (survivors) replace the old population
        return survivors

    def _aco_observer(self, population, num_generations, num_evaluations, args):
        """[INSPYRED] Logs metrics per generation and manages the multimodal archive."""
        valid_pop = [ind for ind in population if ind.fitness != float('inf')]
        if not valid_pop:
            return

        iteration_best_val = min(valid_pop, key=lambda x: x.fitness).fitness
        iteration_avg = sum(ind.fitness for ind in valid_pop) / len(valid_pop)

        best_so_far = args.setdefault('best_so_far', float('inf'))
        if iteration_best_val < best_so_far:
            best_so_far = iteration_best_val
            args['best_so_far'] = best_so_far

        # Log for plots
        self.history["iteration"].append(num_generations)
        self.history["best"].append(best_so_far)
        self.history["iteration_best"].append(iteration_best_val)
        self.history["average"].append(iteration_avg)

        # Manage archive for multimodal alternatives
        quality_threshold = args.get('quality_threshold', 1.05)
        for ind in valid_pop:
            if ind.fitness <= quality_threshold * best_so_far:
                self.archive.append((deepcopy(ind.candidate), ind.fitness))

    # =====================================================================
    # MAIN EXECUTION
    # =====================================================================

    def solve(self, n_alternatives=3, quality_threshold=1.05, min_diversity=0.20):
        # Instantiate the inspyred EC engine
        ea = inspyred.ec.EvolutionaryComputation(self.prng)

        # Inject standard components
        ea.selector = inspyred.ec.selectors.default_selection
        ea.variator = self._aco_variator
        ea.replacer = self._aco_replacer
        ea.observer = self._aco_observer
        ea.terminator = inspyred.ec.terminators.generation_termination

        # Run the evolution
        ea.evolve(
            generator=self._aco_generator,
            evaluator=self._aco_evaluator,
            pop_size=self.n_ants,
            max_generations=self.n_iterations,
            maximize=False,  # We want to minimize distance
            quality_threshold=quality_threshold  # Passed to observer
        )

        # Extract the absolute best solution from our archive
        best_solution = None
        best_cost = float("inf")

        if self.archive:
            self.archive.sort(key=lambda x: x[1])
            best_solution = self.archive[0][0]
            best_cost = self.archive[0][1]

        # Extract diverse multimodal alternatives
        alternatives = []
        for solution, cost in self.archive:
            if len(alternatives) == 0:
                alternatives.append((solution, cost))
                continue

            diverse_enough = all(
                self.diversity(solution, existing_solution) >= min_diversity
                for existing_solution, _ in alternatives
            )

            if diverse_enough:
                alternatives.append((solution, cost))

            if len(alternatives) == n_alternatives:
                break

        return best_solution, best_cost, alternatives

    # =====================================================================
    # PLOTTING FUNCTIONS
    # =====================================================================

    def plot_convergence(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(self.history["iteration"], self.history["best"], label="Global Best")
        plt.plot(self.history["iteration"], self.history["iteration_best"], label="Iteration Best", alpha=0.7)
        plt.plot(self.history["iteration"], self.history["average"], label="Iteration Average", alpha=0.7)

        plt.xlabel("Generation")
        plt.ylabel("Total Distance")
        plt.title("ACO Convergence Over Generations")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def plot_solution(self, solution, title="CVRP Solution"):
        import matplotlib.pyplot as plt

        depot = self.nodes[self.depot_id]

        plt.figure(figsize=(9, 7))

        for route_index, route in enumerate(solution, start=1):
            full_route = [self.depot_id] + route + [self.depot_id]

            x = [self.nodes[node][0] for node in full_route]
            y = [self.nodes[node][1] for node in full_route]

            plt.plot(x, y, marker="o", linewidth=1.5, label=f"Route {route_index}")

        plt.scatter(depot[0], depot[1], marker="s", s=150, label="Depot")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def plot_alternatives(self, alternatives):
        for i, (solution, cost) in enumerate(alternatives, start=1):
            self.plot_solution(
                solution,
                title=f"Alternative {i} | Distance = {cost:.2f}"
            )

    def plot_alternative_comparison(self, best_cost, alternatives):
        import matplotlib.pyplot as plt

        labels = []
        costs = []

        for i, (_, cost) in enumerate(alternatives, start=1):
            labels.append(f"Alt {i}")
            costs.append(cost)

        gaps = [(cost / best_cost - 1) * 100 for cost in costs]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, costs)

        plt.axhline(best_cost, linestyle="--", label=f"Best = {best_cost:.2f}")

        for i, gap in enumerate(gaps):
            plt.text(i, costs[i], f"+{gap:.2f}%", ha="center", va="bottom")

        plt.ylabel("Total Distance")
        plt.title("Comparison of Diverse ACO Alternatives")
        plt.legend()
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.show()