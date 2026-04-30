# Librerias a usar
import math  # Modulo matematico para calcular distancias euclidianas
import numpy as np  # Manejo eficiente de vectores numericos

from inspyred import swarm  # Modulos del motor de inteligencia de enjambre e Inspyred
from typing import Optional  # Para que variables esten a none
from src.pso.base_pso import BasePSO  # Nuestra clase padre
from src.common.problem import CVRPProblem  # Clase que define el problema a resolver


# Clase para la tecnica de Competicion de Especies (SC)
class SpeciesPSO(BasePSO):

    # Constructor que añade los parametros exclusivos de esta tecnica
    def __init__(
        self, datos_problema: CVRPProblem=None,
        radio: float = 0.5,
        penalizacion: Optional[float]=None,
        num_soluciones_corregir=None
    ):
        
        # Invocamos al constructor del padre para inicializar lo basico
        super().__init__(datos_problema, num_soluciones_corregir)
        
        # Asignamos los parametros fisicos de las especies
        self.radio: float = radio
        
        # Calculamos la penalizacion como la distancia maxima existente en el mapa
        self.penalizacion: float = np.max(datos_problema.distance_matrix) if penalizacion is None else penalizacion

    # Funcion que devuelve los parametros de la configuracion
    def get_params_configuracion(self) -> dict:

        # Llamamos la funcion padre
        config_params: dict = super().get_params_configuracion()

        # Añadimos los que falta
        config_params['radio'] = self.radio
        config_params['penalizacion'] = self.penalizacion

        # Lo devolvemos
        return config_params

    # Funcion que evalua los individuos con su fitness
    def evaluator(self, candidates: list, args: dict) -> list:

        # Aplicamos la funcion del padre
        fitness_base: list = super().evaluator(candidates, args)

        # Aplicamos penalizaciones (para obtener multiples soluciones con fitness sharing, clearing, etc)
        fitness_penalizado: np.ndarray = self._aplicar_penalizacion(fitness_base, candidates)

        # Retornamos directamente la lista de distancias sin alterar
        return fitness_penalizado.tolist()

    # Implementamos obligatoriamente la penalizacion (Template Method)
    def _aplicar_penalizacion(self, fitness_base: list, candidatos: list) -> np.ndarray:
        
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