# Librerias a usar
import math  # Modulo matematico para calcular distancias euclidianas
import numpy as np  # Manejo eficiente de vectores numericos

from inspyred import swarm  # Modulos del motor de inteligencia de enjambre e Inspyred
from src.pso.base_pso import BasePSO  # Nuestra clase padre


# Definimos la clase hija exclusiva para la tecnica de Fitness Sharing
class FitnessSharingPSO(BasePSO):

    # Constructor que añade los parametros exclusivos de esta tecnica
    def __init__(
            self, datos_problema,
            radio: float=2.0, alpha: float=1.0,
            num_soluciones_corregir: float=0.05
        ):
        
        # Invocamos al constructor del padre para inicializar lo basico (dimensiones, limites, etc.)
        super().__init__(datos_problema, num_soluciones_corregir)
        
        # Asignamos los parametros fisicos del nicho
        self.radio: float = radio
        self.alpha: float = alpha

    # Funcion que configura el pso dependiendo de los datos
    def config_pso(self, algoritmo: swarm.PSO) -> None:
        
        # Ponemos topologia anillo
        algoritmo.topology = swarm.topologies.ring_topology

    # Funcion para calcular el impacto de la densidad en el fitness
    def _aplicar_penalizacion(self, fitness_base: np.ndarray, candidatos: list) -> np.array:
        
        # Inicializamos el vector de salida
        fitness_final: np.ndarray = np.copy(fitness_base)
        
        # Procesamos cada individuo de la poblacion
        for i, cand1 in enumerate(candidatos):
            
            # Calculamos la suma de distancias normalizadas (densidad)
            densidad_nicho: float = 0.0
            
            # Comparacion contra vecinos
            for j, cand2 in enumerate(candidatos):
                
                # Distancia euclidiana
                distancia: float = math.dist(cand1, cand2)
                
                # Si esta dentro del radio aplica la formula con alpha
                if distancia < self.radio:
                    densidad_nicho += (1.0 - (distancia / self.radio) ** self.alpha)
                    
            # Penalizamos el fitness por la saturacion del nicho
            fitness_final[i] = fitness_base[i] * densidad_nicho
            
        # Retornamos la lista final
        return fitness_final