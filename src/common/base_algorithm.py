# Librerias a usar
import numpy as np  # Manejo eficiente de vectores numericos

from inspyred.ec import Individual  # Para definir el tipado del mejor individuo
from abc import ABC, abstractmethod  # Modulos para crear clases abstractas


# Definimos la clase abstracta padre para todos los algoritmos
class BaseAlgorithm(ABC):
    
    # Constructor que inicializa los parametros comunes del algoritmo
    def __init__(self, tamano_poblacion: int, max_evaluaciones: int):
        
        # Asignamos los parametros fisicos base de la ejecucion
        self.tamano_poblacion: int = tamano_poblacion
        self.max_evaluaciones: int = max_evaluaciones
        
        # Listas donde guardaremos los datos estadisticos para el Visualizador
        self.historico_fitness_mejor: list = []
        self.historico_fitness_media: list = []
        self.historico_diversidad: list = []
        
        # Variable para almacenar las mejores soluciones
        self.mejores_soluciones: list = []

    # Funcion observadora que sirve para almacenar datos como best fitness y mas
    def _observer_fitness_diversidad(self, population: list, num_generations: int, num_evaluations: int, args: dict) -> None:
        
        # Extraemos y guardamos el mejor fitness de esta generacion
        mejor_individuo = max(population)
        self.historico_fitness_mejor.append(mejor_individuo.fitness)
        
        # Calculamos y guardamos el fitness medio de todo el enjambre actual
        media_fitness: float = sum([ind.fitness for ind in population]) / len(population)
        self.historico_fitness_media.append(media_fitness)
        
        # Calculamos la diversidad mediante la desviacion tipica media del genotipo
        diversidad: float = np.array([ind.candidate for ind in population]).std(axis=0).mean()
        self.historico_diversidad.append(diversidad)

    # Funcion para limpiar los datos pasados
    def clear(self) -> None:

        # Limpiamos todo
        self.historico_fitness_mejor: list = []
        self.historico_fitness_media: list = []
        self.historico_diversidad: list = []        
        self.mejores_soluciones: list = []

    # Funcion que ejecuta el algorithmo
    @abstractmethod
    def ejecutar(self, problema) -> None:
        
        # Obligamos a su implementacion en las clases hijas
        pass