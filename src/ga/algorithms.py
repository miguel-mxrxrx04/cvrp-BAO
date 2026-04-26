import random 
from src.common.problem import CVRPProblem

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
    
    def tournament_selection(self, population: list, k: int = 3) -> list:
        """
        Selects the best individual from a random sample of size k.
        """
        best_individual = None
        best_fitness = float('inf')

        tournament = random.sample(population, k)

        for individual in tournament:
            decoded = self.decode_chromosome(individual)
            fitness = self.evaluate_fitness(decoded)

            if fitness < best_fitness:
                best_fitness = fitness
                best_individual = individual
        return best_individual
    
    def order_crossover(self, parent1: list, parent2: list) -> list:
        """
        Performs OX1 crossover, suitable for permutation chromosomes.
        Ensures no duplicate or missing clients in the child.
        """
        size = len(parent1)
        child = [-1] * size

        start, end = sorted(random.sample(range(size), 2))
        child[start:end+1] = parent1[start:end+1]

        p2_pointer = (end + 1) % size
        child_pointer = (end + 1) % size
        
        while -1 in child:
            gene = parent2[p2_pointer]
            if gene not in child:
                child[child_pointer] = gene
                child_pointer = (child_pointer + 1) % size
            p2_pointer = (p2_pointer + 1) % size
        return child

    def swap_mutation(self, chromosome: list, mutation_rate: float = 0.05) -> list:
            """
            Randomly swaps two genes in the chromosome based on the mutation rate.
            Maintains permutation integrity (no duplicate or missing clients).
            """
            # Generamos un número aleatorio entre 0 y 1. Si es menor que el mutation_rate, mutamos.
            if random.random() < mutation_rate:
                # Seleccionamos dos índices distintos al azar
                idx1, idx2 = random.sample(range(len(chromosome)), 2)
                
                # Intercambiamos los valores en esas posiciones
                chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
                
            return chromosome

    def run(self, generations: int = 100, mutation_rate: float = 0.05) -> tuple:
        """
        Executes the main evolutionary loop.
        Returns the best route found, its cost, and the cost history per generation.
        """
        population = self.generate_initial_population()

        best_overall_route = None
        best_overall_cost = float('inf')
        cost_history = []

        print(f"Starting GA for {generations} generations")

        for gen in range(generations):
            new_population = []
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child = self.order_crossover(parent1, parent2)
                child = self.swap_mutation(child, mutation_rate)

                new_population.append(child)
            population = new_population
            current_best_cost = float('inf')
            current_best_individual = None

            for ind in population:
                decoded = self.decode_chromosome(ind)
                cost = self.evaluate_fitness(decoded)

                if cost < current_best_cost:
                    current_best_cost = cost
                    current_best_ind = ind
            
            cost_history.append(current_best_cost)

            if current_best_cost < best_overall_cost:
                best_overall_cost = current_best_cost
                best_overall_route = self.decode_chromosome(current_best_ind)
            
            if gen % 10 == 0 or gen == generations - 1:
                print(f"Generation {gen:3d} | Best cost: {best_overall_cost:.2f}")
        print("Evolution completed.")
        return best_overall_route, best_overall_cost, cost_history



