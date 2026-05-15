import numpy as np
import matplotlib.pyplot as plt


class ACOVisualizer:
    
    def __init__(self, nodes: dict, depot_id: int = 1):
        
        # Store nodes to access coordinates without needing the CVRPProblem class
        self.nodes: dict = nodes
        self.depot_id: int = depot_id
        
        # Palette of 20 colors to differentiate vehicles
        self.palette_colors = plt.cm.tab20.colors

    def plot_evolution(self, best_fitness_history: list, mean_fitness_history: list) -> None:
        
        # Create a single plot since ACO does not track diversity per iteration
        plt.figure(figsize=(8, 5))
        
        # Convergence Curve (Fitness)
        plt.plot(best_fitness_history, color='#2ca02c', linewidth=2, label='Best Fitness')
        plt.plot(mean_fitness_history, color='#1f77b4', linewidth=2, linestyle='--', alpha=0.8, label='Mean fitness')
        
        plt.title('Fitness over Generations (ACO)', fontsize=14, fontweight='bold')
        plt.xlabel('Generation (Iteration)', fontsize=10)
        plt.ylabel('Fitness (Distance)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_population_distribution(self, final_costs: list) -> None:
        
        # Create a canvas with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left Graph: Frequency Histogram
        ax1.hist(
            final_costs,
            bins=15,
            color='#2ca02c',
            edgecolor='black',
            alpha=0.7
        )
        ax1.set_title('Fitness Distribution (Histogram)', fontsize=12)
        ax1.set_xlabel('Route Cost', fontsize=10)
        ax1.set_ylabel('Number of Solutions', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Right Graph: Boxplot for outliers
        ax2.boxplot(
            final_costs,
            vert=False,
            patch_artist=True, 
            boxprops={'facecolor': '#d62728', 'color': 'black'},
            medianprops={'color': 'white', 'linewidth': 2}
        )
        ax2.set_title('Fitness Dispersion (Boxplot)', fontsize=12)
        ax2.set_xlabel('Route Cost', fontsize=10)
        ax2.set_yticks([])
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        fig.suptitle('Statistical Analysis of Final ACO Archive', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    def plot_routes(self, aco_solution: list, title: str = 'Optimized Routes Map') -> None:
        
        # Create a large canvas for the map
        plt.figure(figsize=(12, 8))
        
        # Draw all clients as gray background points
        for node_id, coords in self.nodes.items():
            if node_id != self.depot_id:
                plt.scatter(coords[0], coords[1], c='gray', s=30, alpha=0.5)
                
        # Get and draw the central depot as a large red square
        depot_coord: tuple = self.nodes[self.depot_id]
        plt.scatter(depot_coord[0], depot_coord[1], c='red', marker='s', s=150, label='Central Depot', zorder=5)
        
        # Iterate through the lists of lists (ACO format)
        for num_truck, route in enumerate(aco_solution):
            
            if len(route) > 0:
                
                # We reconstruct the physical path adding the depot at start and end
                full_route: list = [self.depot_id] + route + [self.depot_id]
                
                # Extract X and Y coordinate lists
                x_coords: list = [self.nodes[node][0] for node in full_route]
                y_coords: list = [self.nodes[node][1] for node in full_route]
                
                # Assign cyclic color according to the index
                truck_color: tuple = self.palette_colors[num_truck % len(self.palette_colors)]
                
                # Draw the solid route
                plt.plot(x_coords, y_coords, c=truck_color, linewidth=2, alpha=0.8, marker='o', markersize=5, label=f'Vehicle {num_truck + 1}')
                
        # Final aesthetic configuration of the map
        plt.title(title, fontsize=15, fontweight='bold')
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Move the legend outside the chart to avoid covering paths
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()