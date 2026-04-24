import os

class Parser:
    def __init__(self, filepath: str = ""):
        """Initializes the parser with the .vrp file"""
        self.filepath = filepath
        self.dimension = 0
        self.capacity = 0
        self.nodes = {} # {node_id : (x_coord, y_coord)}
        self.demands = {} # {node_id: demand}
    def parse(self):
        """Reads and extracts key variables"""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        with open(self.filepath, 'r') as file:
            lines = file.readlines()
        
        reading_nodes = False
        reading_demands = False

        for line in lines:
            line = line.strip()

            if not line or line == "EOF":
                continue

            if line.startswith("DIMENSION"):
                self.dimension = int(line.split()[-1])
            elif line.startswith("CAPACITY"):
                self.capacity = int(line.split()[-1])
            
            elif line.startswith("NODE_COORD_SECTION"):
                reading_nodes = True
                reading_demands = False
                continue
            elif line.startswith("DEMAND_SECTION"):
                reading_nodes = False
                reading_demands = True
            elif line.startswith("DEPOT_SECTION"):
                reading_nodes = False
                reading_demands = False
                continue
        
            if reading_nodes:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    self.nodes[node_id] = (float(parts[1]), float(parts[2]))
            elif reading_demands:
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0])
                    self.demands[node_id] = int(parts[1])
        print(f"Parser sucessfully loaded. {self.dimension} nodes. Vehicle capacity: {self.capacity}") 
        return self.nodes, self.demands, self.capacity               
