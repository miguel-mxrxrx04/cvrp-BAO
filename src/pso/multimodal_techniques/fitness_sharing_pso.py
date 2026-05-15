# Librerias a usar
import math  # Modulo matematico para calcular distancias euclidianas
import numpy as np  # Manejo eficiente de vectores numericos

from inspyred import swarm  # Modulos del motor de inteligencia de enjambre e Inspyred
from src.pso.base_pso import BasePSO  # Nuestra clase padre
from src.common.problem import CVRPProblem  # Clase que define el problema a resolver


# Definimos la clase hija exclusiva para la tecnica de Fitness Sharing
class FitnessSharingPSO(BasePSO):

    # Constructor que añade los parametros exclusivos de esta tecnica
    def __init__(
        self,
        datos_problema: CVRPProblem=None,
        radio: float=0.5,
        alpha: float=1.0,
        num_soluciones_corregir=None
    ):
        
        # Invocamos al constructor del padre para inicializar lo basico (dimensiones, limites, etc.)
        super().__init__(datos_problema, num_soluciones_corregir)
        
        # Asignamos los parametros fisicos del nicho
        self.radio: float = self._calcular_radio_dinamico(radio)
        self.alpha: float = alpha

    # Funcion auxiliar para escalar el cráter al tamaño del problema
    def _calcular_radio_dinamico(self, valor_radio: float) -> float:
        
        # Si no hay problema cargado, devolvemos el valor tal cual por seguridad
        if self.datos_problema is None:
            return valor_radio
            
        # Si el radio es mayor o igual a 1.0, asumimos que es una distancia euclídea absoluta
        if valor_radio >= 1.0:
            return valor_radio
            
        # Si el radio es menor a 1.0, se trata como un porcentaje de cobertura (Factor)
        # Obtenemos la dimension
        dimension_real: int = len(self.datos_problema.node_ids) - 1
        
        # Calculamos la diagonal maxima del hipercubo
        distancia_maxima: float = math.sqrt(dimension_real)
        
        # Retornamos la fraccion correspondiente
        return valor_radio * distancia_maxima

    # Funcion que devuelve los parametros de la configuracion
    def get_params_configuracion(self) -> dict:

        # Llamamos la funcion padre
        config_params: dict = super().get_params_configuracion()

        # Añadimos los que falta
        config_params['radio'] = self.radio
        config_params['alpha'] = self.alpha

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

    # Funcion para calcular el impacto de la densidad en el fitness
    def _aplicar_penalizacion(self, fitness_base: list, candidatos: list) -> np.array:
        
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