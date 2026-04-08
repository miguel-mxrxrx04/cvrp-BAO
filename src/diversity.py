class DiversityHandler:
    @staticmethod
    def get_edges(solution: list) -> set:
        """
        Converts a solution route [0, 1, 2, 0, 3, 4, 0] into a set of unique edges.
        Example output: {(0,1), (1,2), (0,2), (0,3), (3,4), (0,4)}
        """
        edges = set()
        for i in range(len(solution) - 1):
            # Store the edge as a sorted tuple to ignore the direction of travel
            # e.g., traveling 1 -> 2 is structurally the same edge as 2 -> 1
            edge = tuple(sorted((solution[i], solution[i+1])))
            edges.add(edge)
        return edges

    @staticmethod
    def calculate_jaccard_distance(sol1: list, sol2: list) -> float:
        """
        Calculates the Jaccard distance between two solutions based on their edges.
        Returns a float between 0.0 and 1.0:
        - 0.0 means the routes are structurally identical.
        - 1.0 means the routes share absolutely no edges.
        """
        edges1 = DiversityHandler.get_edges(sol1)
        edges2 = DiversityHandler.get_edges(sol2)
        
        intersection = len(edges1.intersection(edges2))
        union = len(edges1.union(edges2))
        
        # Prevent division by zero edge-case
        if union == 0:
            return 0.0
            
        # Jaccard Distance = 1 - (Intersection / Union)
        return 1.0 - (intersection / union)
