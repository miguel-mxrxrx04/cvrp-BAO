# Librerias a usar
import math  # Modulo matematico para calcular distancias euclidianas
import numpy as np  # Manejo eficiente de vectores numericos

from inspyred import swarm  # Modulos del motor de inteligencia de enjambre e Inspyred
from src.pso.base_pso import BasePSO  # Nuestra clase padre


# Clase para la tecnica de Competicion de Especies (SC)
class SpeciesPSO(BasePSO):

    # Constructor que añade los parametros exclusivos de esta tecnica
    def __init__(
            self, datos_problema, radio: float = 0.5,
            penalizacion: float=float('inf'),
            num_soluciones_corregir: float = 0.05
        ):
        
        # Invocamos al constructor del padre para inicializar lo basico
        super().__init__(datos_problema, num_soluciones_corregir)
        
        # Asignamos los parametros fisicos de las especies
        self.radio: float = radio
        self.penalizacion: float = penalizacion

    # Funcion que configura el pso dependiendo de los datos
    def config_pso(self, algoritmo: swarm.PSO) -> None:
        
        # Ponemos topologia anillo
        algoritmo.topology = swarm.topologies.ring_topology

    # Implementamos obligatoriamente la penalizacion (Template Method)
    def _aplicar_penalizacion(self, fitness_base: np.ndarray, candidatos: list) -> np.ndarray:
        
        # Inicializamos el vector de salida copiando el original
        fitness_final: np.ndarray = np.copy(fitness_base)
        
        # Obtenemos los indices ordenados de mejor a peor fitness (para dar prioridad a los alfas)
        indices_ordenados: np.ndarray = np.argsort(fitness_base)
        
        # Lista para almacenar los indices de las particulas que son "Semillas" (lideres de especie)
        semillas_especie: list = []
        
        # Iteramos en orden de calidad para identificar a los lideres
        for i in indices_ordenados:
            
            # Variable para marcar si el individuo pertenece a una especie ya consolidada
            es_de_especie_existente: bool = False
            
            # Comparamos al individuo actual contra TODAS las semillas ya identificadas
            for idx_semilla in semillas_especie:
                
                # Calculamos la distancia fisica contra la semilla
                distancia_a_semilla: float = math.dist(candidatos[i], candidatos[idx_semilla])
                
                # Si cae dentro del territorio de una semilla, le pertenece a su especie
                if distancia_a_semilla < self.radio:
                    
                    # Lo marcamos como subdito de esa especie
                    es_de_especie_existente = True
                    
                    # Dejamos de buscar, ya sabemos a que especie pertenece
                    break
                    
            # Si terminamos de mirar y no pertenece a nadie, ¡ha descubierto un nuevo territorio!
            if not es_de_especie_existente:
                
                # Lo guardamos como un nuevo lider de especie
                semillas_especie.append(i)
                
            # Si por el contrario pertenece a una especie y no es el lider original...
            else:
                
                # Aplicamos el castigo por ser una copia debil del lider
                fitness_final[i] += self.penalizacion
                
        # Retornamos el array de numpy alterado para que el padre finalice el proceso
        return fitness_final