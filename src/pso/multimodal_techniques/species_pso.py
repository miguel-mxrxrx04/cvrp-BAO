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
        radio: float=0.5,
        penalizacion: Optional[float]=None,
        num_soluciones_corregir=None
    ):
        
        # Invocamos al constructor del padre para inicializar lo basico
        super().__init__(datos_problema, num_soluciones_corregir)
        
        # Asignamos los parametros fisicos de las especies
        self.radio: float = self._calcular_radio_dinamico(radio)
        
        # Calculamos la penalizacion como la distancia maxima existente en el mapa
        if self.datos_problema is not None:
            self.penalizacion: float = 2 * np.sum(self.datos_problema.distance_matrix[0, 1:]) if penalizacion is None else penalizacion
        else:
            self.penalizacion: float = 0.0 if penalizacion is None else penalizacion

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

    # Funcion que evalua los individuos con su fitness
    def evaluator(self, candidates: list, args: dict) -> list:

        # Aplicamos la funcion del padre
        fitness_base: list = super().evaluator(candidates, args)

        # Aplicamos penalizaciones (para obtener multiples soluciones con fitness sharing, clearing, etc)
        fitness_penalizado: np.ndarray = self._aplicar_penalizacion(fitness_base, candidates)

        # Retornamos directamente la lista de distancias sin alterar
        return fitness_penalizado.tolist()

    # Funcion para penalizar (SC)
    def _aplicar_penalizacion(self, fitness_base: list, candidatos: list) -> np.ndarray:
        
        # Inicializamos el vector de salida copiando el original
        fitness_final: np.ndarray = np.copy(fitness_base)
        
        # Obtenemos los indices ordenados de mejor a peor fitness (para dar prioridad a los alfas)
        indices_ordenados: np.ndarray = np.argsort(fitness_base)
        
        # Lista para almacenar los indices de las particulas que son "Semillas" (lideres de especie)
        semillas_especie: list = []

        # Diccionario para saber a que especie pertenece cada particula
        asignacion_especie: dict = {}
        
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

                # Lo ponemos en su propia especie
                asignacion_especie[i] = i
                
            # Si no lo es
            else:
                
                # Le ponemos a que especia pertenece
                asignacion_especie[i] = idx_semilla

        # Penalización SC basada en saturación de especie
        for idx, semilla in asignacion_especie.items():

            # Vemos si no es la semilla
            if idx not in semillas_especie:

                # Cuántos individuos hay en esta especie
                tamanio_especie = list(asignacion_especie.values()).count(semilla)

                # Penalización: especies grandes = fitness peor
                fitness_final[idx] += self.penalizacion * (tamanio_especie / len(candidatos))
                
        # Retornamos el nuevo fitness
        return fitness_final