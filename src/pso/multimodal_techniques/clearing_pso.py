# Librerias a usar
import math  # Modulo matematico para calcular distancias euclidianas
import numpy as np  # Manejo eficiente de vectores numericos

from inspyred import swarm  # Modulos del motor de inteligencia de enjambre e Inspyred
from typing import Optional  # Para que variables esten a none
from src.pso.base_pso import BasePSO  # Nuestra clase padre
from src.common.problem import CVRPProblem  # Clase que define el problema a resolver


# Clase para applicar la tecnica de Clearing
class ClearingPSO(BasePSO):

    # Constructor que añade los parametros exclusivos de esta tecnica
    def __init__(
        self,
        datos_problema: CVRPProblem=None,
        radio: float=0.5,
        penalizacion: Optional[float]=None,
        num_soluciones_corregir=None
    ):
        
        # Invocamos al constructor del padre para inicializar lo basico
        super().__init__(datos_problema, num_soluciones_corregir)
        
        # Asignamos los parametros fisicos de la limpieza
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

    # Funcion que sirve para aplicar penalizaciones al fitness (FS)
    def _aplicar_penalizacion(self, fitness_base: list, candidatos: list) -> np.ndarray:
        
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