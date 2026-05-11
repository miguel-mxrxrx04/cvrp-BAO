import random 
import inspyred
from src.common.problem import CVRPProblem
from src.common.diversity import DiversityHandler

class MultimodalGeneticAlgorithm:
    def __init__(self, problem: CVRPProblem, pop_size: int = 100) -> None:
        """
        Initializes the GA for multimodal CVRP
        """
        self.problem = problem
        self.pop_size = pop_size

        self.clients = [
            node_id for node_id in self.problem.node_ids
            if node_id != self.problem.depot_id
        ]
        # We'll store the history here via the observer
        self.cost_history = []

    # --- INSPYRED COMPONENTS ---
    def evaluate_population(self, candidates: list, args: dict) -> list:
        """
        Calculates fitness for the entire population.
        """
        fitness_values = []
        for candidate in candidates:
            decoded_route = self.decode_chromosome(candidate)
            fitness = self.problem.evaluate_route_distance(decoded_route)
            fitness_values.append(fitness)
        return fitness_values

    def inversion_mutation(self, prng: random.Random, candidate: list, mutation_rate: float) -> list:
        """Double Inversion to forcefully break out of deep local optima."""
        mutated = candidate[:]
        if prng.random() < mutation_rate:
            if len(mutated) >= 4:
                # Primera inversión
                idx1, idx2 = sorted(prng.sample(range(len(mutated)), 2))
                mutated[idx1:idx2] = reversed(mutated[idx1:idx2])
                # Segunda inversión (el golpe de gracia al estancamiento)
                idx3, idx4 = sorted(prng.sample(range(len(mutated)), 2))
                mutated[idx3:idx4] = reversed(mutated[idx3:idx4])
        return mutated
    
    def generate_chromosome(self, random, args: dict) -> list:
        """
        Generates a chromosome using a Randomized Nearest Neighbor heuristic.
        Massively improves the starting quality (Generation 0) to compete with ACO.
        """
        # 100% random initial route 85% of times
        if random.random() < 0.85:
            chromosome = list(self.clients)
            random.shuffle(chromosome)
            return chromosome

        unvisited = set(self.clients)
        chromosome = []
        
        # Start off in random node
        current_node = random.choice(list(unvisited))
        unvisited.remove(current_node)
        chromosome.append(current_node)
        
        # Looking for nearby nodes, heuristic
        while unvisited:
            # Seleccionamos un "vecindario" de los 5 nodos más cercanos al actual
            # (Esto añade aleatoriedad controlada, a diferencia de un Greedy puro)
            candidates = random.sample(list(unvisited), min(5, len(unvisited)))
            
            # De esos 5, elegimos el que esté más cerca físicamente
            best_candidate = None
            min_dist = float('inf')
            
            # Necesitamos las coordenadas para medir (asumimos que las tienes en self.problem.nodes)
            curr_coords = self.problem.nodes[current_node]
            
            for candidate in candidates:
                cand_coords = self.problem.nodes[candidate]
                # Distancia euclidiana simple al cuadrado (más rápido de computar)
                dist_sq = (curr_coords[0] - cand_coords[0])**2 + (curr_coords[1] - cand_coords[1])**2
                
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    best_candidate = candidate
                    
            current_node = best_candidate
            unvisited.remove(current_node)
            chromosome.append(current_node)
            
        return chromosome
    
    def custom_variator(self, random: random.Random, candidates: list, args: dict) -> list:
        """
        Inspyred Variator: Applies OX1 crossover, INVERSION mutation, and optional 2-opt Local Search.
        """
        mutation_rate = args.setdefault('mutation_rate', 0.20)
        use_local_search = args.setdefault('use_local_search', False)
        # Aseguramos que la variable coincida con el método run()
        ls_prob = args.setdefault('local_search_prob', 0.15) 

        offspring = []
        for i in range(0, len(candidates), 2):
            parent1 = candidates[i]
            parent2 = candidates[i+1] if i + 1 < len(candidates) else candidates[0]

            child1 = self.order_crossover(parent1, parent2)
            child2 = self.order_crossover(parent2, parent1)

            # inversion instead of swap
            child1 = self.inversion_mutation(random, child1, mutation_rate)
            child2 = self.inversion_mutation(random, child2, mutation_rate)

            # probability control for the memetic
            if use_local_search:
                if random.random() < ls_prob:
                    child1 = self.apply_windowed_2opt(child1)
                if random.random() < ls_prob:
                    child2 = self.apply_windowed_2opt(child2)

            offspring.extend([child1, child2])

        return offspring[:len(candidates)]
    
    def diversity_replacer(self, random, population, offspring, args, **kwargs):
        """
        Custom Inspyred Replacer (Crowding): 
        Ensures the population maintains structural diversity during evolution.
        """
        threshold = args.setdefault('similarity_threshold', 0.15)
        combined = population + offspring
        # Sort by fitness (lowest is best)
        combined.sort(key=lambda x: x.fitness) 
        
        unique_population = []
        for individual in combined:
            if len(unique_population) >= self.pop_size:
                break
                
            is_diverse = True
            route_ind = self.decode_chromosome(individual.candidate)
            
            for accepted in unique_population:
                route_acc = self.decode_chromosome(accepted.candidate)
                
                # Usamos la clase común para medir la diversidad
                dist = DiversityHandler.calculate_jaccard_distance(route_ind, route_acc)
                
                if dist < threshold: # Too similar to an existing one
                    is_diverse = False
                    break
            
            if is_diverse:
                unique_population.append(individual)
        
        # If we filtered out too many, fill the rest with the remaining best individuals
        if len(unique_population) < self.pop_size:
            for ind in combined:
                if ind not in unique_population:
                    unique_population.append(ind)
                    if len(unique_population) >= self.pop_size:
                        break
                        
        return unique_population
       
    def observer_tracker(self, population: list, num_generations: int, num_evaluations: int, args: dict) -> None:
        """
        Inspyred Observer: Runs at the end of each generation to track progress.
        """
        best_fitness = min([ind.fitness for ind in population])
        self.cost_history.append(best_fitness)
        
        if num_generations % 10 == 0:
            print(f"Generation {num_generations:3d} | Best cost: {best_fitness:.2f}")

    # --- DOMAIN LOGIC ---
    def decode_chromosome(self, chromosome: list) -> list:
        """
        Converts a client permutation into a valid CVRP route with depots.
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
        """Calculates total distance."""
        return self.problem.evaluate_route_distance(decoded_route)
    
    def order_crossover(self, parent1: list, parent2: list) -> list:
        """Performs OX1 crossover."""
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
        """Performs swap mutation."""
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome
    
    def apply_windowed_2opt(self, chromosome: list, max_iterations: int = 3, window_size: int = 100) -> list:
        """
        Enhanced 2-opt Local Search. 
        Window size expanded to allow global untangling of random initial routes.
        """
        best_chromosome = chromosome[:]
        best_cost = self.evaluate_fitness(self.decode_chromosome(best_chromosome))
        
        improved = True
        iterations = 0
        chrom_len = len(chromosome)
        
        while improved and iterations < max_iterations:
            improved = False
            for i in range(chrom_len - 1):
                # Ampliamos el límite para revisar casi toda la ruta
                limit = min(i + window_size, chrom_len) 
                for j in range(i + 2, limit):
                    new_chrom = best_chromosome[:i] + best_chromosome[i:j][::-1] + best_chromosome[j:]
                    new_cost = self.evaluate_fitness(self.decode_chromosome(new_chrom))
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_chromosome = new_chrom
                        improved = True
                        break # Encontramos mejora, aplicamos y reiniciamos búsqueda
                if improved:
                    break
            iterations += 1
            
        return best_chromosome

    def run(self, generations: int = 100, mutation_rate: float = 0.20,  
            similarity_threshold: float = 0.15, use_local_search: bool = False,
            tournament_size: int = 2, local_search_prob: float = 0.15) -> tuple:
        """
        Orchestrates the Inspyred engine and extracts the Top 3 distinct routes.
        'use_local_search' acts as a toggle between Pure GA and Memetic GA.
        """
        algo_type = "Memetic" if use_local_search else "Pure"
        print(f"Starting {algo_type} Multimodal GA for {generations} generations.")
        
        prng = random.Random()
        ga_engine = inspyred.ec.EvolutionaryComputation(prng)

        ga_engine.selector = inspyred.ec.selectors.tournament_selection
        ga_engine.replacer = self.diversity_replacer 
        ga_engine.variator = self.custom_variator
        ga_engine.terminator = inspyred.ec.terminators.generation_termination
        ga_engine.observer = self.observer_tracker

        self.cost_history = []

        # Execute the evolution
        final_population = ga_engine.evolve(
            generator=self.generate_chromosome,
            evaluator=self.evaluate_population,
            pop_size=self.pop_size,
            bounder=inspyred.ec.DiscreteBounder(self.clients),
            maximize=False, 
            max_generations=generations,
            tournament_size=tournament_size,          # CORREGIDO: Dinámico, no hardcodeado a 3
            mutation_rate=mutation_rate,
            similarity_threshold=similarity_threshold,
            use_local_search=use_local_search,
            local_search_prob=local_search_prob       # NUEVO: Frena la homogeneización del 2-opt
        )

        final_population.sort(key=lambda x: x.fitness)
        
        top_3_solutions = []
        for ind in final_population:
            if len(top_3_solutions) >= 3:
                break
                
            route = self.decode_chromosome(ind.candidate)
            cost = ind.fitness
            
            is_novel = True
            for accepted_route, _ in top_3_solutions:
                if DiversityHandler.calculate_jaccard_distance(route, accepted_route) < similarity_threshold:
                    is_novel = False
                    break
                    
            if is_novel:
                top_3_solutions.append((route, cost))

        if len(top_3_solutions) < 3:
            for ind in final_population:
                if len(top_3_solutions) >= 3:
                    break
                route = self.decode_chromosome(ind.candidate)
                if not any(route == acc_route for acc_route, _ in top_3_solutions):
                    top_3_solutions.append((route, ind.fitness))

        print("Evolution completed.")
        return top_3_solutions, self.cost_history