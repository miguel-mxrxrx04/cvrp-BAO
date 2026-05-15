# Librerias a usar
import math  # Modulo matematico 
import numpy as np  # Manejo eficiente de vectores numericos

from inspyred import swarm  # Necesario para cambiar la topologia a estrella
from typing import Optional  # Para que variables esten a none
from src.pso.base_pso import BasePSO  # Nuestra clase padre abstracta
from src.common.problem import CVRPProblem  # Clase que define el problema a resolver
from src.common.diversity import DiversityHandler  # Para comparar rutas (distancia Jaccard)


# Definimos la clase hija exclusiva para la tecnica Secuencial (El Mapa de Crateres)
class SequentialPSO(BasePSO):

    # Constructor que añade los parametros exclusivos de esta tecnica
    def __init__(
        self,
        datos_problema: CVRPProblem=None,
        radio: float=0.5,
        penalizacion: Optional[float]=None,
        num_soluciones_corregir= None
    ):
        
        # Invocamos al constructor del padre para inicializar lo basico
        super().__init__(datos_problema, num_soluciones_corregir)
        
        # Asignamos el radio
        self.radio: float = self._calcular_radio_dinamico(radio)

        # Calculamos la penalizacion como la distancia maxima existente en el mapa
        if self.datos_problema is not None:
            self.penalizacion: float = 2 * np.sum(self.datos_problema.distance_matrix[0, 1:]) if penalizacion is None else penalizacion
        else:
            self.penalizacion: float = 0.0 if penalizacion is None else penalizacion

        # Atributo para guardar al mejor de la iteracion (servira para coger el mejor de la ultima iteracion y asi hacer el SN)
        self.mejor_local: Optional[list] = None
        
        # Lista con las RUTAS DECODIFICADAS optimas encontradas en iteraciones anteriores
        self.optimos_encontrados: list = []

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
        config_params['penalizacion'] = self.penalizacion

        # Lo devolvemos
        return config_params

    # Funcion que configura el pso dependiendo de los datos
    def config_pso(self, algoritmo: swarm.PSO) -> None:
        
        # Lo ponemos en estrella porque sacaremos el mejor (evitamos que solo se comparta informacion con los vecinos mas cercanos)
        algoritmo.topology = swarm.topologies.star_topology

        # Vemos si no es la primera iteracion
        if self.mejor_local is not None:

            # Añadimos LA RUTA del mejor de la ejecucion al archivo historico de cráteres
            self.optimos_encontrados.append(self.mejor_local)

    # Funcion que evalua los individuos con su fitness
    def evaluator(self, candidates: list, args: dict) -> list:

        # Aplicamos la funcion del padre
        fitness_base: list = super().evaluator(candidates, args)

        # Aplicamos penalizaciones (para obtener multiples soluciones con fitness sharing, clearing, etc)
        fitness_penalizado: np.ndarray = self._aplicar_penalizacion(fitness_base, candidates)

        # Retornamos directamente la lista de distancias sin alterar
        return fitness_penalizado.tolist()

    # Funcion para aplicar la penalizacion (SN)
    def _aplicar_penalizacion(self, fitness_base: list, candidatos: list) -> np.ndarray:
        
        # Inicializamos el vector de salida copiando el original
        fitness_final: np.ndarray = np.copy(fitness_base)
        
        # Comprobamos cada particula actual contra el archivo historico de rutas prohibidas
        for i, cand in enumerate(candidatos):
            
            # Iteramos por los vectores optimos descubiertos en ejecuciones anteriores
            for vector_optimo in self.optimos_encontrados:
                
                # Calculamos la distancia fisica entre ambos vectores en el hiperespacio
                similitud: float = math.dist(cand, vector_optimo)
                
                # Si la particula intenta explorar una ruta muy parecida a la ya conquistada
                if similitud <= self.radio:
                    
                    # Arruinamos su fitness sumandole la penalizacion radiactiva
                    fitness_final[i] += self.penalizacion
                    
                    # Como ya ha sido castigada, dejamos de comparar contra otras rutas
                    break

        # Guardamos el mejor de esta iteracion
        indice_mejor: int = int(np.argmin(fitness_final))
        self.mejor_local = candidatos[indice_mejor]
                    
        # Retornamos el array de numpy alterado
        return fitness_final