import math
import random
from copy import deepcopy


class ACOSolver:
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

        random.seed(seed)

        self.customers = [i for i in nodes if i != depot_id]
        self.dist = self._build_distance_matrix()
        self.pheromone = {
            (i, j): 1.0 for i in nodes for j in nodes if i != j
        }
        self.history = {
        "iteration": [],
        "best": [],
        "iteration_best": [],
        "average": [],
        }

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
            return random.choice(feasible)

        r = random.random() * total
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

    def _deposit_pheromones(self, solution):
        cost = self.solution_distance(solution)
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

    def solve(self, n_alternatives=3, quality_threshold=1.05, min_diversity=0.20):
        best_solution = None
        best_cost = float("inf")
        archive = []

        for iteration in range(self.n_iterations):
            iteration_solutions = []

            for _ in range(self.n_ants):
                solution = self.construct_solution()
                solution = self.improve_solution(solution)

                if not self.is_valid(solution):
                    continue

                cost = self.solution_distance(solution)
                iteration_solutions.append((solution, cost))

                if cost < best_cost:
                    best_solution = deepcopy(solution)
                    best_cost = cost

            self._evaporate_pheromones()

            if iteration_solutions:
                iteration_solutions.sort(key=lambda x: x[1])
                self._deposit_pheromones(iteration_solutions[0][0])
                iteration_best = iteration_solutions[0][1]
                iteration_avg = sum(cost for _, cost in iteration_solutions) / len(iteration_solutions)

                self.history["iteration"].append(iteration)
                self.history["best"].append(best_cost)
                self.history["iteration_best"].append(iteration_best)
                self.history["average"].append(iteration_avg)

            for solution, cost in iteration_solutions:
                if cost <= quality_threshold * best_cost:
                    archive.append((deepcopy(solution), cost))

        archive.sort(key=lambda x: x[1])

        alternatives = []

        for solution, cost in archive:
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