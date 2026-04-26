import numpy as np
import math

class CVRPProblem:
    def __init__(self, nodes: dict, demands: dict, capacity: int):
        """Initializes the CVRP environment and precalculates distances."""
        self.nodes = nodes
        self.demands = demands
        self.capacity = capacity
        self.dimension = len(nodes)

        self.node_ids = list(self.nodes.keys())
        self.depot_id = self.node_ids[0]

        self.distance_matrix = self._calculate_distance_matrix()
        print(f"[Problem] Distance matrix ({self.dimension}x{self.dimension}) precalculated.")
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculates the Eucliden distance matrix between all nodes."""
        matrix = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            for j in range(self.dimension):
                id_i = self.node_ids[i]
                id_j = self.node_ids[j]

                x1, y1 = self.nodes[id_i]
                x2, y2 = self.nodes[id_j]

                matrix[i][j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return matrix

    def get_distance(self, init_pos: int, target_pos: int):
        """Returns the euclidean distance between 2 positions"""
        return self.distance_matrix[init_pos][target_pos]
    
    def evaluate_route_distance(self, route: list) -> float:
        """
        Calculates the total distance of a full solution.
        Expected format: [1, 3, 4, 1, 2, 1] where 1 is the depot.
        """
        total_distance = 0.0
        for i in range(len(route) - 1):
            idx_from = self.node_ids.index(route[i])
            idx_to = self.node_ids.index(route[i+1])
            total_distance += self.distance_matrix[idx_from][idx_to]
        return total_distance
    
    def is_route_valid(self, route: list) ->  bool:
        """Checks if the solution strictly meets the vehicle capacity constraint."""
        current_load = 0
        for node_id in route:
            if node_id == self.depot_id:
                current_load = 0 # Vehicle empty when starting new sub-route
            else:
                current_load += self.demands[node_id]
                if current_load > self.capacity:
                    return False # Capacity exceeded
        return True