import os
import re
import time
import shutil
import random
import copy
import numpy as np
import pandas as pd
import concurrent.futures

from pathlib import Path
from src.common.parser import Parser
from src.aco.ACO import ACOSolver

class ACOExperimentExecuter:
    def __init__(self, data_path: str):
        script_path = Path(__file__).resolve()
        self.data_path = str(script_path.parents[3] / data_path)

        divided_problems = self._divide_problems()

        self.small_problems = set(divided_problems[0])
        self.medium_problems = set(divided_problems[1])
        self.large_problems = set(divided_problems[2])

        self.num_problems = len(self.small_problems) + len(self.medium_problems) + len(self.large_problems)

    def _divide_problems(self) -> tuple:
        folder_files = os.listdir(self.data_path)
        vrp_instances = [f for f in folder_files if f.endswith('.vrp')]

        small_problems = []
        medium_problems = []
        large_problems = []

        pattern = re.compile(r'n(\d+)-k')

        for file_name in vrp_instances:
            match = pattern.search(file_name)
            
            if match:
                nodes = int(match.group(1))
                if nodes < 300:
                    small_problems.append(file_name)
                elif nodes < 750:
                    medium_problems.append(file_name)
                else:
                    large_problems.append(file_name)

        return small_problems, medium_problems, large_problems

    def _calculate_params(self, n_customers: int) -> dict:
        params = {}

        if n_customers <= 50:
            params['n_ants'] = 10
            params['n_iterations'] = 250
        elif n_customers <= 300:
            params['n_ants'] = 20
            params['n_iterations'] = 50
        elif n_customers <= 750:
            params['n_ants'] = 35
            params['n_iterations'] = 75
        else:
            params['n_ants'] = 50
            params['n_iterations'] = 100
            
        return params

    def run_single_experiment(
        self,
        problem_name: str,
        n_ants: int,
        n_iterations: int,
        constraint_handling: str,
        multimodal_technique: str,
        seed: int,
        alpha: float=1.0,
        beta: float=3.0, 
        evaporation: float=0.25,
        min_diversity: float=0.20,
        n_solutions: int=3
    ) -> tuple:
        
        problem_path = os.path.join(self.data_path, problem_name)
        parser = Parser(problem_path)
        nodes, demands, capacity = parser.parse()

        aco_solver = ACOSolver(
            nodes=nodes,
            demands=demands,
            capacity=capacity,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            evaporation=evaporation,
            constraint_handling=constraint_handling,
            multimodal_technique=multimodal_technique,
            seed=seed
        )

        _, _, alternatives = aco_solver.solve(n_alternatives=n_solutions, min_diversity=min_diversity)

        return aco_solver, alternatives

    def _run_single_repetition_worker(
        self,
        run_id: int,
        problem_name: str,
        n_ants: int,
        n_iterations: int,
        n_solutions: int,
        constraint_handling: str,
        multimodal_technique: str,
        alpha: float,
        beta: float,
        evaporation: float,
        min_diversity: float
    ) -> dict:
        
        print(f'Processing ACO experiment: {problem_name} - run {run_id}')

        new_seed = random.SystemRandom().randint(0, 9999999) + run_id
        start_time = time.time()

        aco_solver, alternatives = self.run_single_experiment(
            problem_name, 
            n_ants, 
            n_iterations,
            constraint_handling, 
            multimodal_technique, 
            new_seed,
            alpha,
            beta,
            evaporation,
            min_diversity,
            n_solutions
        )

        execution_time = time.time() - start_time

        costs = [cost for _, cost in alternatives]
        mean_fitness = float(np.mean(costs)) if costs else 0.0
        std_fitness = float(np.std(costs)) if len(costs) > 1 else 0.0

        jaccard_distances = []
        if len(alternatives) > 1:
            for i in range(len(alternatives)):
                for j in range(i + 1, len(alternatives)):
                    dist = aco_solver.diversity(alternatives[i][0], alternatives[j][0])
                    jaccard_distances.append(dist)
                    
        mean_jaccard = float(np.mean(jaccard_distances)) if jaccard_distances else 0.0

        return {
            'run': run_id,
            'mean_fitness': mean_fitness,
            'std_fitness': std_fitness,
            'mean_jaccard_distance': mean_jaccard,
            'n_evaluations': aco_solver.n_ants * aco_solver.n_iterations,
            'n_generations': aco_solver.n_iterations,
            'experiment_id': problem_name.split('.')[0],
            'tiempo': execution_time
        }

    def _run_repeated_parallel(self, **kwargs) -> pd.DataFrame:
        n_repeat = kwargs['n_repeat']
        statistical_results = []
        safe_threads = max(1, os.cpu_count() - 3)

        alpha = kwargs.get('alpha', 1.0)
        beta = kwargs.get('beta', 3.0)
        evaporation = kwargs.get('evaporation', 0.25)
        min_diversity = kwargs.get('min_diversity', 0.20)

        with concurrent.futures.ProcessPoolExecutor(max_workers=safe_threads) as executor:
            futures = []
            for run in range(n_repeat):
                future = executor.submit(
                    self._run_single_repetition_worker,
                    run,
                    kwargs['problem_name'],
                    kwargs['n_ants'],
                    kwargs['n_iterations'],
                    kwargs.get('n_solutions', 3),
                    kwargs['constraint_handling'],
                    kwargs['multimodal_technique'],
                    alpha, beta, evaporation, min_diversity
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    statistical_results.append(future.result())
                except Exception as e:
                    print(f"Critical error in parallel execution: {e}")

        df_results = pd.DataFrame(statistical_results)
        return df_results.sort_values(by=['run']).reset_index(drop=True)

    def _run_repeated_sequential(self, **kwargs) -> pd.DataFrame:
        statistical_results = []

        alpha = kwargs.get('alpha', 1.0)
        beta = kwargs.get('beta', 3.0)
        evaporation = kwargs.get('evaporation', 0.25)
        min_diversity = kwargs.get('min_diversity', 0.20)
        
        for run in range(kwargs['n_repeat']):
            row_result = self._run_single_repetition_worker(
                run,
                kwargs['problem_name'],
                kwargs['n_ants'],
                kwargs['n_iterations'],
                kwargs.get('n_solutions', 3),
                kwargs['constraint_handling'],
                kwargs['multimodal_technique'],
                alpha, beta, evaporation, min_diversity
            )
            statistical_results.append(row_result)
            
        return pd.DataFrame(statistical_results)

    def run_repeated_experiment(self, **kwargs) -> pd.DataFrame:
        if kwargs.get('do_parallel', False):
            return self._run_repeated_parallel(**kwargs)
        else:
            return self._run_repeated_sequential(**kwargs)

    def run_all_experiments(
        self,
        experiment_folder: str,
        constraint_handling: str = "feasibility",
        multimodal_technique: str = "archive",
        n_repeat: int = 35,
        do_parallel: bool = False,
        overwrite: bool = False,
        **kwargs
    ) -> None:
        
        processed_files = set()

        if os.path.isdir(experiment_folder):
            if overwrite:
                shutil.rmtree(experiment_folder)
            else:
                self._get_solved_problems(processed_files, experiment_folder)

        vrp_instances = self._get_problems_to_process(processed_files)
        os.makedirs(experiment_folder, exist_ok=True)

        if not vrp_instances:
            print('No problems to process in this execution.')
            return

        for instance in vrp_instances:
            total_customers = self._get_num_customers(instance)
            scaled_params = self._calculate_params(total_customers)
            
            print(f'Starting tests for: {instance} ({total_customers} customers) | Ants: {scaled_params["n_ants"]} | Iter: {scaled_params["n_iterations"]}')

            df_results = self.run_repeated_experiment(
                problem_name=instance,
                n_ants=scaled_params['n_ants'],
                n_iterations=scaled_params['n_iterations'],
                constraint_handling=constraint_handling,
                multimodal_technique=multimodal_technique,
                n_repeat=n_repeat,
                do_parallel=do_parallel,
                **kwargs
            )

            clean_name = instance.split('.')[0]
            export_path = os.path.join(experiment_folder, f'experiment_{clean_name}.csv')
            df_results.to_csv(export_path, index=False)

    def _get_solved_problems(self, processed_files: set, experiment_folder: str) -> None:
        csv_pattern = re.compile(r'experiment_(.*)\.csv')
        
        for file_name in os.listdir(experiment_folder):
            match = csv_pattern.match(file_name)
            if match:
                processed_files.add(f'{match.group(1)}.vrp')

    def _get_problems_to_process(self, processed_files: set) -> list:
        small = set(self.small_problems - processed_files)
        medium = set(self.medium_problems - processed_files)
        large = set(self.large_problems - processed_files)
        
        return list(small | medium | large)

    def _get_num_customers(self, problem_name: str) -> int:
        problem_path = os.path.join(self.data_path, problem_name)
        temp_parser = Parser(problem_path)
        nodes, _, _ = temp_parser.parse()
        
        return len(nodes) - 1