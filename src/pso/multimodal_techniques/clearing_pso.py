# Librerias a usar
import math  # Modulo matematico para calcular distancias euclidianas
import numpy as np  # Manejo eficiente de vectores numericos

from inspyred import swarm  # Modulos del motor de inteligencia de enjambre e Inspyred
from src.pso.base_pso import BasePSO  # Nuestra clase padre


# Clase para applicar la tecnica de Clearing
class ClearingPSO(BasePSO):

    # Constructor que añade los parametros exclusivos de esta tecnica
    def __init__(
            self, datos_problema, radio: float=0.5,
            penalizacion: float=float('inf'),
            num_soluciones_corregir: float=0.05
        ):
        
        # Invocamos al constructor del padre para inicializar lo basico
        super().__init__(datos_problema, num_soluciones_corregir)
        
        # Asignamos los parametros fisicos de la limpieza
        self.radio: float = radio
        self.penalizacion: float = penalizacion

    # Funcion que configura el pso dependiendo de los datos
    def config_pso(self, algoritmo: swarm.PSO) -> None:
        
        # Ponemos topologia anillo
        algoritmo.topology = swarm.topologies.ring_topology

    # Implementamos obligatoriamente la penalizacion (Template Method)
    def _aplicar_penalizacion(self, fitness_base: np.ndarray, candidatos: list) -> np.ndarray:
        
        # Inicializamos el vector de salida
        fitness_final: np.ndarray = np.copy(fitness_base)
        
        # Obtenemos el orden de calidad de los individuos (de mejor a peor)
        indices: np.ndarray = np.argsort(fitness_base)
        
        # Marcador booleano de individuos que deben ser ignorados (aniquilados)
        muertos: np.ndarray = np.zeros(len(candidatos), dtype=bool)
        
        # Iteramos protegiendo a los mejores de cada radio
        for i in indices:
            
            # Si este individuo ya fue limpiado por un lider mejor, no puede ser lider
            if muertos[i]:
                continue
                
            # Revisamos al resto de individuos peores que el
            for j in indices:
                
                # Comprobamos distancia contra el lider actual (si no es el mismo y sigue vivo)
                if i != j and not muertos[j] and math.dist(candidatos[i], candidatos[j]) < self.radio:
                    
                    # Sumamos el castigo gigantesco para arruinar su probabilidad de ser elegido
                    fitness_final[j] += self.penalizacion
                    
                    # Lo marcamos como muerto para no evaluarlo como lider en futuras iteraciones
                    muertos[j] = True
                    
        # Retornamos el array de numpy alterado para que el padre siga el proceso
        return fitness_final