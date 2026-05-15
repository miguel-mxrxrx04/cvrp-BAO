import math
import random 
from typing import List, Dict, Tuple, Set, Any
import inspyred
import time

from src.common.problem import CVRPProblem
from src.common.diversity import DiversityHandler

class MultimodalGeneticAlgorithm:
    def __init__(self, problem: CVRPProblem, pop_size: int = 100) -> None:
        """
        Initializes the Genetic Algorithm for multimodal CVRP.
        
        Args:
            problem (CVRPProblem): The specific CVRP instance to solve.
            pop_size (int): Number of individuals in the population.
        """
        self.problem: CVRPProblem = problem
        self.pop_size: int = pop_size

        # Extract client IDs (excluding the depot)
        self.clients: List[int] = [
            node_id for node_id in self.problem.node_ids
            if node_id != self.problem.depot_id
        ]
        
        # History tracker populated by the observer during evolution
        self.cost_history: List[float] = []
        
        # Pre-compute distances to optimize performance
        self.dist_matrix: Dict[Tuple[int, int], float] = self._build_distance_matrix()

    def _build_distance_matrix(self) -> Dict[Tuple[int, int], float]:
        """
        Pre-calculates the Euclidean distance between all nodes.
        
        Returns:
            Dict[Tuple[int, int], float]: A dictionary mapping node pairs to distances.
        """
        dist: Dict[Tuple[int, int], float] = {}
        for i in self.problem.nodes:
            for j in self.problem.nodes:
                if i != j:
                    x1, y1 = self.problem.nodes[i]
                    x2, y2 = self.problem.nodes[j]
                    dist[(i, j)] = math.hypot(x1 - x2, y1 - y2)
        return dist

    # --- INSPYRED COMPONENTS ---

    def evaluate_population(self, candidates: List[List[int]], args: Dict[str, Any]) -> List[float]:
        """
        Calculates fitness (total route distance) for the entire population.
        
        Args:
            candidates (List[List[int]]): The list of chromosomes to evaluate.
            args (Dict[str, Any]): Additional arguments passed by Inspyred.
            
        Returns:
            List[float]: The fitness values corresponding to each candidate.
        """
        fitness_values: List[float] = []
        for candidate in candidates:
            decoded_route: List[int] = self.decode_chromosome(candidate)
            fitness: float = self.problem.evaluate_route_distance(decoded_route)
            fitness_values.append(fitness)
        return fitness_values

    def inversion_mutation(self, prng: random.Random, candidate: List[int], mutation_rate: float) -> List[int]:
        """
        Applies a Double Inversion mutation to forcefully break out of deep local optima.
        
        Args:
            prng (random.Random): The pseudo-random number generator instance.
            candidate (List[int]): The chromosome to mutate.
            mutation_rate (float): Probability of mutation occurring.
            
        Returns:
            List[int]: The potentially mutated chromosome.
        """
        mutated: List[int] = candidate[:]
        if prng.random() < mutation_rate:
            if len(mutated) >= 4:
                # First inversion segment
                idx1, idx2 = sorted(prng.sample(range(len(mutated)), 2))
                mutated[idx1:idx2] = list(reversed(mutated[idx1:idx2]))
                # Second inversion segment to ensure significant structural change
                idx3, idx4 = sorted(prng.sample(range(len(mutated)), 2))
                mutated[idx3:idx4] = list(reversed(mutated[idx3:idx4]))
        return mutated
    
    def generate_chromosome(self, random: random.Random, args: Dict[str, Any]) -> List[int]:
        """
        Generates a chromosome using a Cluster-First, Route-Second Heuristic.
        This mirrors the successful approach used in the PSO implementation, grouping
        nodes by spatial proximity and capacity before generating the final sequence.
        
        Args:
            random (random.Random): Inspyred's internal random number generator.
            args (Dict[str, Any]): Additional arguments passed by Inspyred.
            
        Returns:
            List[int]: A newly generated chromosome.
        """
        # Maintain 20% pure randomness to preserve baseline genetic diversity
        if random.random() < 0.20:
            chromosome: List[int] = list(self.clients)
            random.shuffle(chromosome)
            return chromosome

        # --- Cluster-First Grouping Heuristic ---
        unvisited: Set[int] = set(self.clients)
        chromosome: List[int] = []
        
        while unvisited:
            # 1. Start a new "vehicle" (cluster) from the depot
            current_node: int = self.problem.depot_id
            current_load: int = 0
            
            # 2. Fill the vehicle with the nearest valid nodes until capacity is reached
            while unvisited:
                # Find valid candidates that do not exceed vehicle capacity
                valid_candidates: List[int] = [
                    c for c in unvisited 
                    if current_load + self.problem.demands[c] <= self.problem.capacity
                ]
                
                if not valid_candidates:
                    break # Vehicle is full or remaining nodes exceed capacity; close cluster
                
                # Sort valid candidates by proximity to the current node using cached distances
                valid_candidates.sort(key=lambda c: self.dist_matrix[(current_node, c)])
                
                # Select randomly from the top 3 closest nodes to maintain stochasticity
                top_candidates: List[int] = valid_candidates[:3]
                best_candidate: int = random.choice(top_candidates)
                
                # Append to the route sequence and update states
                chromosome.append(best_candidate)
                unvisited.remove(best_candidate)
                current_load += self.problem.demands[best_candidate]
                current_node = best_candidate

        return chromosome
    
    def custom_variator(self, random: random.Random, candidates: List[inspyred.ec.Individual], args: Dict[str, Any]) -> List[List[int]]:
        """
        Inspyred Variator: Orchestrates OX1 crossover, INVERSION mutation, 
        and optional Windowed 2-opt Local Search (Memetic refinement).
        
        Args:
            random (random.Random): Inspyred's internal random number generator.
            candidates (List[inspyred.ec.Individual]): The parent population selected for reproduction.
            args (Dict[str, Any]): Dictionary containing algorithmic parameters.
            
        Returns:
            List[List[int]]: The generated offspring population.
        """
        mutation_rate: float = args.setdefault('mutation_rate', 0.20)
        use_local_search: bool = args.setdefault('use_local_search', False)
        ls_prob: float = args.setdefault('local_search_prob', 0.15) 

        offspring: List[List[int]] = []
        for i in range(0, len(candidates), 2):
            parent1: List[int] = candidates[i]
            # Handle odd population sizes safely
            parent2: List[int] = candidates[i+1] if i + 1 < len(candidates) else candidates[0]

            # Apply Order Crossover (OX1)
            child1: List[int] = self.order_crossover(parent1, parent2)
            child2: List[int] = self.order_crossover(parent2, parent1)

            # Apply Double Inversion Mutation
            child1 = self.inversion_mutation(random, child1, mutation_rate)
            child2 = self.inversion_mutation(random, child2, mutation_rate)

            # Apply Memetic Local Search (if enabled and probability hits)
            if use_local_search:
                if random.random() < ls_prob:
                    child1 = self.apply_windowed_2opt(child1)
                if random.random() < ls_prob:
                    child2 = self.apply_windowed_2opt(child2)

            offspring.extend([child1, child2])

        # Ensure we return exactly the requested number of offspring
        return offspring[:len(candidates)]
    
    def diversity_replacer(self, random: random.Random, population: List[inspyred.ec.Individual], 
                           offspring: List[inspyred.ec.Individual], args: Dict[str, Any], **kwargs: Any) -> List[inspyred.ec.Individual]:
        """
        Custom Inspyred Replacer (Crowding Mechanism): 
        Ensures the survivor population maintains a minimum structural diversity threshold.
        
        Args:
            random (random.Random): Inspyred's internal RNG.
            population (List[Individual]): The previous generation's population.
            offspring (List[Individual]): The newly created offspring.
            args (Dict[str, Any]): Dictionary containing algorithmic parameters.
            
        Returns:
            List[Individual]: The selected survivor population for the next generation.
        """
        threshold: float = args.setdefault('similarity_threshold', 0.15)
        combined: List[inspyred.ec.Individual] = population + offspring
        
        # Sort combined population by best fitness (lowest distance)
        combined.sort(key=lambda x: x.fitness) 
        
        unique_population: List[inspyred.ec.Individual] = []
        for individual in combined:
            if len(unique_population) >= self.pop_size:
                break
                
            is_diverse: bool = True
            route_ind: List[int] = self.decode_chromosome(individual.candidate)
            
            # Check against currently accepted diverse individuals
            for accepted in unique_population:
                route_acc: List[int] = self.decode_chromosome(accepted.candidate)
                dist: float = DiversityHandler.calculate_jaccard_distance(route_ind, route_acc)
                
                # Reject if topological similarity is below the acceptable threshold
                if dist < threshold: 
                    is_diverse = False
                    break
            
            if is_diverse:
                unique_population.append(individual)
        
        # Fallback: If strict diversity filtering leaves the population undersized, 
        # fill the remainder with the best available non-selected individuals.
        if len(unique_population) < self.pop_size:
            for ind in combined:
                if ind not in unique_population:
                    unique_population.append(ind)
                    if len(unique_population) >= self.pop_size:
                        break
                        
        return unique_population
       
    def observer_tracker(self, population: List[inspyred.ec.Individual], num_generations: int, num_evaluations: int, args: Dict[str, Any]) -> None:
        """Inspyred Observer: Executes at the end of each generation."""
        best_fitness: float = min([ind.fitness for ind in population])
        self.cost_history.append(best_fitness)

    # --- DOMAIN LOGIC ---

    def decode_chromosome(self, chromosome: List[int]) -> List[int]:
        """
        Converts a linear client permutation into a valid CVRP route containing depot visits.
        
        Args:
            chromosome (List[int]): The sequence of clients.
            
        Returns:
            List[int]: The full route including necessary depot returns for capacity constraints.
        """
        route: List[int] = [self.problem.depot_id]
        current_load: int = 0

        for client in chromosome:
            demand: int = self.problem.demands[client]
            # If adding this client exceeds capacity, return to depot first
            if current_load + demand > self.problem.capacity:
                route.append(self.problem.depot_id)
                current_load = 0
            
            route.append(client)
            current_load += demand
        
        # Ensure the final route terminates at the depot
        if route[-1] != self.problem.depot_id:
            route.append(self.problem.depot_id)
        
        return route
    
    def evaluate_fitness(self, decoded_route: List[int]) -> float:
        """
        Wrapper to calculate the total Euclidean distance of a fully decoded CVRP route.
        """
        return self.problem.evaluate_route_distance(decoded_route)
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Performs Order Crossover (OX1), preserving relative ordering and absolute positions.
        
        Args:
            parent1 (List[int]): Primary parent chromosome.
            parent2 (List[int]): Secondary parent chromosome.
            
        Returns:
            List[int]: The generated offspring chromosome.
        """
        size: int = len(parent1)
        child: List[int] = [-1] * size

        # Select a random continuous segment from parent1
        start, end = sorted(random.sample(range(size), 2))
        child[start:end+1] = parent1[start:end+1]

        # Fill the remaining genes from parent2, preserving their relative order
        p2_pointer: int = (end + 1) % size
        child_pointer: int = (end + 1) % size
        
        while -1 in child:
            gene: int = parent2[p2_pointer]
            if gene not in child:
                child[child_pointer] = gene
                child_pointer = (child_pointer + 1) % size
            p2_pointer = (p2_pointer + 1) % size
            
        return child

    def apply_windowed_2opt(self, chromosome: List[int], max_iterations: int = 3, window_size: int = 100) -> List[int]:
        """
        Enhanced 2-opt Local Search. 
        Explores localized edge reversals to untangle routes and eliminate cross-overs.
        
        Args:
            chromosome (List[int]): The base chromosome to optimize.
            max_iterations (int): Maximum optimization passes.
            window_size (int): Lookahead limit for edge swaps to control computational overhead.
            
        Returns:
            List[int]: The locally optimized chromosome.
        """
        best_chromosome: List[int] = chromosome[:]
        best_cost: float = self.evaluate_fitness(self.decode_chromosome(best_chromosome))
        
        improved: bool = True
        iterations: int = 0
        chrom_len: int = len(chromosome)
        
        while improved and iterations < max_iterations:
            improved = False
            for i in range(chrom_len - 1):
                limit: int = min(i + window_size, chrom_len) 
                for j in range(i + 2, limit):
                    # Perform edge swap via sub-route reversal
                    new_chrom: List[int] = best_chromosome[:i] + best_chromosome[i:j][::-1] + best_chromosome[j:]
                    new_cost: float = self.evaluate_fitness(self.decode_chromosome(new_chrom))
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_chromosome = new_chrom
                        improved = True
                        break # Break inner loop upon finding an improvement
                if improved:
                    break # Break outer loop to restart search from the newly optimized state
            iterations += 1
            
        return best_chromosome

    def run(self, generations: int = 100, mutation_rate: float = 0.20,  
            similarity_threshold: float = 0.15, use_local_search: bool = False,
            tournament_size: int = 2, local_search_prob: float = 0.15) -> Tuple[List[Tuple[List[int], float]], List[float]]:
        """
        Orchestrates the Inspyred evolutionary engine and extracts diverse routing niches.
        
        Args:
            generations (int): Maximum number of generations to run.
            mutation_rate (float): Probability of mutation occurring.
            similarity_threshold (float): Minimum required topological diversity between niches.
            use_local_search (bool): Toggles the Memetic 2-opt refinement phase.
            tournament_size (int): Size of the tournament selection bracket.
            local_search_prob (float): Probability of applying 2-opt to an individual.
            
        Returns:
            Tuple: A list of the top 3 diverse niches (route, cost) and the historical convergence data.
        """
        algo_type: str = "Memetic" if use_local_search else "Pure"
        print(f"Starting {algo_type} Multimodal GA for {generations} generations.")
        
        prng: random.Random = random.Random()
        ga_engine: inspyred.ec.EvolutionaryComputation = inspyred.ec.EvolutionaryComputation(prng)

        # Configure evolutionary architecture
        ga_engine.selector = inspyred.ec.selectors.tournament_selection
        ga_engine.replacer = self.diversity_replacer 
        ga_engine.variator = self.custom_variator
        ga_engine.terminator = inspyred.ec.terminators.generation_termination
        ga_engine.observer = self.observer_tracker

        self.cost_history.clear()

        # Execute the primary evolution loop
        final_population: List[inspyred.ec.Individual] = ga_engine.evolve(
            generator=self.generate_chromosome,
            evaluator=self.evaluate_population,
            pop_size=self.pop_size,
            bounder=inspyred.ec.DiscreteBounder(self.clients),
            maximize=False, 
            max_generations=generations,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            similarity_threshold=similarity_threshold,
            use_local_search=use_local_search,
            local_search_prob=local_search_prob
        )

        # Sort the final population strictly by fitness
        final_population.sort(key=lambda x: x.fitness)
        
        # Extract the Top 3 structurally distinct solutions (Niches)
        top_3_solutions: List[Tuple[List[int], float]] = []
        
        for ind in final_population:
            if len(top_3_solutions) >= 3:
                break
                
            route: List[int] = self.decode_chromosome(ind.candidate)
            cost: float = ind.fitness
            
            is_novel: bool = True
            for accepted_route, _ in top_3_solutions:
                if DiversityHandler.calculate_jaccard_distance(route, accepted_route) < similarity_threshold:
                    is_novel = False
                    break
                    
            if is_novel:
                top_3_solutions.append((route, cost))

        # Fallback processing if rigid diversity constraints yield fewer than 3 niches
        if len(top_3_solutions) < 3:
            for ind in final_population:
                if len(top_3_solutions) >= 3:
                    break
                route = self.decode_chromosome(ind.candidate)
                # Check for strict topological equality rather than distance threshold
                if not any(route == acc_route for acc_route, _ in top_3_solutions):
                    top_3_solutions.append((route, ind.fitness))

        print("Evolution completed.")
        return top_3_solutions, self.cost_history
    

class PassiveArchiveGA(MultimodalGeneticAlgorithm):
    def __init__(self, problem: Any, pop_size: int = 100) -> None:
        super().__init__(problem, pop_size)
        self.passive_archive: List[Tuple[List[int], float]] = []
        
    def observer_tracker(self, population: List[inspyred.ec.Individual], num_generations: int, num_evaluations: int, args: Dict[str, Any]) -> None:
        best_fitness: float = min([ind.fitness for ind in population])
        self.cost_history.append(best_fitness)
        
        threshold_fitness: float = best_fitness * 1.10 
        
        for ind in population:
            if ind.fitness <= threshold_fitness:
                route: List[int] = self.decode_chromosome(ind.candidate)
                self.passive_archive.append((route, ind.fitness))
                
        if num_generations % 20 == 0:
            print(f"  [Log] Gen {num_generations:3d} | Best: {best_fitness:.2f} | Archive Size: {len(self.passive_archive)}")

    def run(self, generations: int = 150, mutation_rate: float = 0.20, similarity_threshold: float = 0.20, local_search_prob: float = 0.80) -> Tuple[List[Tuple[List[int], float]], List[float]]:
        prng: random.Random = random.Random()
        ga_engine: inspyred.ec.EvolutionaryComputation = inspyred.ec.EvolutionaryComputation(prng)

        ga_engine.selector = inspyred.ec.selectors.tournament_selection
        ga_engine.replacer = inspyred.ec.replacers.generational_replacement 
        ga_engine.variator = self.custom_variator
        ga_engine.terminator = inspyred.ec.terminators.generation_termination
        ga_engine.observer = self.observer_tracker

        self.cost_history.clear()
        self.passive_archive.clear()

        ga_engine.evolve(
            generator=self.generate_chromosome, 
            evaluator=self.evaluate_population, 
            pop_size=self.pop_size,
            bounder=inspyred.ec.DiscreteBounder(self.clients), 
            maximize=False, 
            max_generations=generations,
            tournament_size=3, 
            mutation_rate=mutation_rate, 
            use_local_search=True,
            local_search_prob=local_search_prob
        )

        self.passive_archive.sort(key=lambda x: x[1])
        filtered_niches: List[Tuple[List[int], float]] = []
        
        for route, cost in self.passive_archive:
            if len(filtered_niches) >= 3:
                break
            
            is_novel: bool = True
            for accepted_route, _ in filtered_niches:
                if DiversityHandler.calculate_jaccard_distance(route, accepted_route) < similarity_threshold:
                    is_novel = False
                    break
                    
            if is_novel:
                filtered_niches.append((route, cost))

        return filtered_niches, self.cost_history
    
class SequentialNichingGA(MultimodalGeneticAlgorithm):
    def __init__(self, problem: Any, pop_size: int = 100) -> None:
        super().__init__(problem, pop_size)
        self.tabu_niches: List[List[int]] = [] # Memory of structurally forbidden routes
        self.history_sequential: List[float] = []

    def evaluate_population(self, candidates: List[Any], args: Dict[str, Any]) -> List[float]:
        """
        Evaluates fitness but applies a massive penalty (derating function) 
        to routes topologically similar to already discovered (Tabu) niches.
        """
        fitness_values: List[float] = []
        penalty_factor: float = 10.0 # Multiply cost by 10 to kill it genetically
        similarity_threshold: float = args.setdefault('similarity_threshold', 0.20)

        for candidate in candidates:
            decoded_route: List[int] = self.decode_chromosome(candidate)
            base_fitness: float = self.problem.evaluate_route_distance(decoded_route)

            is_tabu: bool = False
            for tabu_route in self.tabu_niches:
                # If it's too similar to a known niche, mark as tabu
                if DiversityHandler.calculate_jaccard_distance(decoded_route, tabu_route) < similarity_threshold:
                    is_tabu = True
                    break

            if is_tabu:
                fitness_values.append(base_fitness * penalty_factor)
            else:
                fitness_values.append(base_fitness)

        return fitness_values

    def run_sequential(self, num_niches: int = 3, generations_per_niche: int = 80, 
                       mutation_rate: float = 0.20, similarity_threshold: float = 0.20, 
                       local_search_prob: float = 0.80) -> Tuple[List[Tuple[List[int], float]], List[float]]:
        
        extracted_niches: List[Tuple[List[int], float]] = []
        start_time: float = time.perf_counter()

        for interval in range(num_niches):
            print(f"  --> Running Sequential Interval {interval+1}/{num_niches} ({generations_per_niche} generations)...")
            
            prng: random.Random = random.Random()
            ga_engine: inspyred.ec.EvolutionaryComputation = inspyred.ec.EvolutionaryComputation(prng)

            ga_engine.selector = inspyred.ec.selectors.tournament_selection
            # Generational replacement is safe here because the penalty function actively handles diversity
            ga_engine.replacer = inspyred.ec.replacers.generational_replacement 
            ga_engine.variator = self.custom_variator
            ga_engine.terminator = inspyred.ec.terminators.generation_termination
            
            self.cost_history.clear()
            ga_engine.observer = self.observer_tracker

            final_pop: List[Any] = ga_engine.evolve(
                generator=self.generate_chromosome,
                evaluator=self.evaluate_population,
                pop_size=self.pop_size,
                bounder=inspyred.ec.DiscreteBounder(self.clients),
                maximize=False,
                max_generations=generations_per_niche,
                tournament_size=3,
                mutation_rate=mutation_rate,
                similarity_threshold=similarity_threshold,
                use_local_search=True,
                local_search_prob=local_search_prob # CRITICAL FIX: Repaired the Memetic link
            )

            # Extract the best individual from this sequential interval
            final_pop.sort(key=lambda x: x.fitness)
            best_ind: Any = final_pop[0]
            best_route: List[int] = self.decode_chromosome(best_ind.candidate)
            
            # Recalculate physical distance (ignoring artificial derating penalties)
            real_cost: float = self.problem.evaluate_route_distance(best_route)
            
            extracted_niches.append((best_route, real_cost))
            
            # CRITICAL: Append to Tabu list to force exploration elsewhere in the next interval
            self.tabu_niches.append(best_route)
            
            # Store continuous history across all intervals
            self.history_sequential.extend(self.cost_history)
            
        execution_time: float = time.perf_counter() - start_time
        print(f"\n✅ Sequential Extraction Complete in {execution_time:.2f} seconds.")
        return extracted_niches, self.history_sequential

    
class SequentialPenaltyGA(SequentialNichingGA):
    
    def decode_chromosome(self, chromosome: List[int]) -> List[int]:
        """
        SOFT CONSTRAINTS: We allow the vehicle to overload up to 150% 
        before forcing a return to depot.
        """
        route: List[int] = [self.problem.depot_id]
        current_load: int = 0

        for client in chromosome:
            demand: int = self.problem.demands[client]
            if current_load + demand > self.problem.capacity * 1.5:
                route.append(self.problem.depot_id)
                current_load = 0
            
            route.append(client)
            current_load += demand
        
        if route[-1] != self.problem.depot_id:
            route.append(self.problem.depot_id)
        
        return route

    def evaluate_population(self, candidates: List[Any], args: Dict[str, Any]) -> List[float]:
        fitness_values: List[float] = []
        tabu_penalty_factor: float = 10.0 
        capacity_penalty_weight: float = 50.0 
        similarity_threshold: float = args.setdefault('similarity_threshold', 0.20)

        for candidate in candidates:
            decoded_route: List[int] = self.decode_chromosome(candidate)
            base_fitness: float = self.problem.evaluate_route_distance(decoded_route)

            # --- 1. Penalty for Overloading ---
            overload_penalty: float = 0.0
            current_load: int = 0
            for node in decoded_route:
                if node == self.problem.depot_id:
                    if current_load > self.problem.capacity:
                        overload_penalty += (current_load - self.problem.capacity) * capacity_penalty_weight
                    current_load = 0
                else:
                    current_load += self.problem.demands[node]

            # --- 2. Penalty for Tabu Similarity ---
            is_tabu: bool = False
            for tabu_route in self.tabu_niches:
                if DiversityHandler.calculate_jaccard_distance(decoded_route, tabu_route) < similarity_threshold:
                    is_tabu = True
                    break

            final_fitness: float = base_fitness + overload_penalty
            if is_tabu:
                final_fitness *= tabu_penalty_factor
                
            fitness_values.append(final_fitness)

        return fitness_values