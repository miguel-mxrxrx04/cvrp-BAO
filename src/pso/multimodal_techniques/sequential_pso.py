# Librerias a usar
import math  # Modulo matematico 
import numpy as np  # Manejo eficiente de vectores numericos

from inspyred import swarm  # Necesario para cambiar la topologia a estrella
from typing import Optional  # Para que variables esten a none
from src.pso.base_pso import BasePSO  # Nuestra clase padre abstracta
from src.common.problem import CVRPProblem  # Clase que define el problema a resolver


# Definimos la clase hija exclusiva para la tecnica Secuencial (El Mapa de Crateres)
class SequentialPSO(BasePSO):

    # Constructor que añade los parametros exclusivos de esta tecnica
    def __init__(
        self,
        datos_problema: CVRPProblem=None,
        umbral_similitud: float=0.8,
        penalizacion: Optional[float]=None,
        num_soluciones_corregir= None
    ):
        
        # Invocamos al constructor del padre para inicializar lo basico
        super().__init__(datos_problema, num_soluciones_corregir)
        
        # Asignamos el limite de similitud (ej. 0.8 significa 80% de calles iguales) y el castigo
        self.umbral_similitud: float = umbral_similitud

        # Calculamos la penalizacion como la distancia maxima existente en el mapa
        self.penalizacion: float = np.max(datos_problema.distance_matrix) if penalizacion is None else penalizacion

        # Tupla para guardar (fitness, ruta_decodificada) de la mejor solucion local
        self.mejor_local: Optional[tuple] = None
        
        # Lista con las RUTAS DECODIFICADAS optimas encontradas en iteraciones anteriores
        self.optimos_encontrados: list = []

    # Funcion que devuelve los parametros de la configuracion
    def get_params_configuracion(self) -> dict:

        # Llamamos la funcion padre
        config_params: dict = super().get_params_configuracion()

        # Añadimos los que falta
        config_params['umbral_similitud'] = self.umbral_similitud
        config_params['penalizacion'] = self.penalizacion

        # Lo devolvemos
        return config_params

    # Funcion que configura el pso dependiendo de los datos
    def config_pso(self, algoritmo: swarm.PSO) -> None:
        
        # Lo ponemos en estrella porque sacaremos el mejor (evitamos que solo se comparta informacion con los vecinos mas cercanos)
        algoritmo.topology = swarm.topologies.star_topology

        # Vemos si hemos ejecutado ya el algoritmo (tenemos un mejor local guardado)
        if self.mejor_local is not None:

            # Añadimos LA RUTA del mejor de la ejecucion al archivo historico de cráteres
            self.optimos_encontrados.append(self.mejor_local[1])

        # Reiniciamos el rastreador local para la siguiente iteracion del enjambre
        self.mejor_local = None

    # Funcion que evalua los individuos con su fitness
    def evaluator(self, candidates: list, args: dict) -> list:

        # Aplicamos la funcion del padre
        fitness_base: list = super().evaluator(candidates, args)

        # Aplicamos penalizaciones (para obtener multiples soluciones con fitness sharing, clearing, etc)
        fitness_penalizado: np.ndarray = self._aplicar_penalizacion(fitness_base, candidates)

        # Retornamos directamente la lista de distancias sin alterar
        return fitness_penalizado.tolist()

    # Implementamos obligatoriamente la penalizacion comparando calles reales en vez de decimales
    def _aplicar_penalizacion(self, fitness_base: list, candidatos: list) -> np.ndarray:
        
        # Inicializamos el vector de salida copiando el original
        fitness_final: np.ndarray = np.copy(fitness_base)
        
        # Comprobamos cada particula actual contra el archivo historico de rutas prohibidas
        for i, cand in enumerate(candidatos):
            
            # Decodificamos la particula continua para ver que ruta fisica va a hacer
            ruta_cand: list = self.decodificar_spv(cand)
            
            # Iteramos por las rutas optimas descubiertas en vuelos anteriores
            for ruta_optima in self.optimos_encontrados:
                
                # Calculamos que porcentaje de calles comparten ambas rutas
                similitud: float = self._calcular_similitud_rutas(ruta_cand, ruta_optima)
                
                # Si la particula intenta explorar una ruta muy parecida a la ya conquistada
                if similitud >= self.umbral_similitud:
                    
                    # Arruinamos su fitness sumandole la penalizacion radiactiva
                    fitness_final[i] += self.penalizacion
                    
                    # Como ya ha sido castigada, dejamos de comparar contra otras rutas
                    break

        # Nos guardamos el mejor de esta generacion (despues de penalizar)
        self._actualizar_mejor_local(fitness_final, candidatos)
                    
        # Retornamos el array de numpy alterado
        return fitness_final

    # Funcion auxiliar para calcular la similitud estructural entre dos rutas (Basado en Indice de Jaccard)
    def _calcular_similitud_rutas(self, ruta_1: list, ruta_2: list) -> float:
        
        # Extraemos las calles de la ruta 1 ordenando los pares para ignorar el sentido de la marcha (A->B es igual a B->A)
        calles_1: set = {tuple(sorted((ruta_1[i], ruta_1[i+1]))) for i in range(len(ruta_1) - 1)}
        
        # Extraemos las calles de la ruta 2 con la misma logica bidireccional
        calles_2: set = {tuple(sorted((ruta_2[i], ruta_2[i+1]))) for i in range(len(ruta_2) - 1)}
        
        # Encontramos la interseccion (cuantas calles fisicas comparten exactamente)
        interseccion: int = len(calles_1.intersection(calles_2))
        
        # Encontramos la union (el total de calles unicas entre ambas rutas)
        union: int = len(calles_1.union(calles_2))
        
        # Prevenimos el error de division por cero por si llegaran rutas vacias
        if union == 0:
            return 0.0
            
        # Retornamos el Indice de Similitud de Jaccard (1.0 = identicas, 0.0 = totalmente distintas)
        return interseccion / union

    # Funcion que actualiza el mejor local guardando su ruta fisica para no recalcularla despues
    def _actualizar_mejor_local(self, fitness_final: np.ndarray, candidatos: list):

        # Sacamos el indice y el fitness del mejor candidato de esta generacion
        indice_mejor: int = int(np.argmin(fitness_final))
        mejor_fitness_actual: float = fitness_final[indice_mejor]

        # Si el radar esta vacio, o encontramos uno con mejor nota (menor fitness)
        if self.mejor_local is None or mejor_fitness_actual < self.mejor_local[0]:
            
            # Extraemos el candidato vectorial ganador
            mejor_candidato_actual: list = candidatos[indice_mejor]
            
            # Lo decodificamos a ruta fisica al instante para guardarlo
            mejor_ruta_fisica: list = self.decodificar_spv(mejor_candidato_actual)
            
            # Sobreescribimos el mejor local con la nueva tupla (fitness, ruta_fisica)
            self.mejor_local = (mejor_fitness_actual, mejor_ruta_fisica)