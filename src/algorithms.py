import random 
from src.problem import CVRPProblem

class MultimodalGeneticAlgorithm:
    def __init__(self, problem: CVRPProblem, pop_size: int = 100):
        """
        Initializes the GA for multimodal CVRP
        """
        self.problem = problem
        self.pop_size = pop_size

        self.clients = [
            node_id for node_id in self.problem.node_ids
            if node_id != self.problem.depot_id
        ]

    def generate_initial_population(self) -> list:
        """
        Generates the initial population.
        Each individual is a random permutation of cliente IDs.
        """
        population = []
        for _ in range(self.pop_size):
            individual = self.clients.copy()
            random.shuffle(individual)
            population.append(individual)
        print(f"Initial population of {self.pop_size} individuals generated.")
        return population
    
    def decode_chromosome(self, chromosome: list) -> list:
        """
        Converts a giant client permuatition into a valid CVRP route.
        Inserts the depot at the start, end, and whenever capacity is exceeded.
        """
        route = [self.problem.depot_id]
        current_load = 0

        for client in chromosome:
            demand = self.problem.demands[client]
            if current_load + demand > self.problem.capacity:
                route.append(self.problem.depot_id)
                current_load = 0
            route.append(client)
            current_load += demand
        
        if route[-1] != self.problem.depot_id:
            route.append(self.problem.depot_id)
        
        return route
    
    def evaluate_fitness(self, decoded_route: list) -> float:
        """
        Calculates the total distance of a valid, decoded route.
        Lower fitness is better (minimization problem)
        """
        return self.problem.evaluate_route_distance(decoded_route)
    

