# Libreria a usar
import numpy as np  # Para obtener funciones matematicas y de arrays mas optimas

from src.common.problem import CVRPProblem  # Clase padre


# Definimos la clase del problema para un clúster específico, heredando del problema base
class ClusterCVRPProblem(CVRPProblem):
    
    # Constructor que inicializa el problema e inyecta el limite de camiones
    def __init__(self, nodes: dict, demands: dict, capacity: int, m_minimo: int, truck_penalty: int):
        
        # Invocamos al constructor de la clase padre para configurar distancias
        super().__init__(nodes, demands, capacity)
        
        # Asignamos el limite teorico de camiones calculado para esta zona
        self.m_minimo: int = m_minimo

        # Asignamos la penalizacion por pasarnos de camiones
        self.truck_penalty: int = truck_penalty

        # Precalculamos el alpha de penalizacion (distancia maxima)
        self.alpha_penalizacion: float = np.max(self.distance_matrix)

    # FUncion para aplicar la penalizacion en el fitness (cada cluster tendra un minimo ideal de camiones para resolver su sub-problema)
    def evaluate_route_distance(self, route: list) -> float:
        
        # Obtenemos la distancia fisica pura usando el metodo original del padre
        distancia_fisica: float = super().evaluate_route_distance(route)
        
        # Calculamos cuantos camiones ha usado REALMENTE esta particula (apariciones del deposito - 1)
        camiones_usados: int = route.count(self.depot_id) - 1
        
        # Calculamos el exceso de camiones frente al minimo teorico estipulado
        exceso: int = max(0, camiones_usados - self.m_minimo) ** self.truck_penalty
        
        # Penalizamos la distancia multiplicandola por el castigo cuadratico del exceso
        fitness_penalizado: float = distancia_fisica + (self.alpha_penalizacion * exceso)
        
        # Retornamos el fitness final alterado
        return fitness_penalizado