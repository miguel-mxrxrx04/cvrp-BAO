# Librerias a usar
import random  # Modulo para la generacion de numeros pseudoaleatorios

from inspyred import swarm, ec  # Modulos del motor de inteligencia de enjambre e Inspyred
from src.pso.base_pso import BasePSO  # Clase para tipado
from src.pso.pso_algorithm import PSOAlgorithm  # Heredaremos de este cambiando solo la ejecucion


# CLase que resuelve el problema mediante nichos secuenciales
class SequentialPSOAlgorithm(PSOAlgorithm):
    
    def __init__(
        self,
        tamano_poblacion: int,
        max_evaluaciones: int,
        inercia: float=0.7,
        cognitivo: float=1.5,
        social: float=1.5,
        tamano_vecindario: int=5,
        semilla: int=42,
        num_ejecuciones: int=3,
        verbose: bool = False
    ):
        
        super().__init__(
            tamano_poblacion,
            max_evaluaciones,
            inercia,
            cognitivo,
            social,
            tamano_vecindario,
            semilla,
            verbose
        )

        # Definimos el numero de ejecuciones
        self.num_ejecuciones: int = num_ejecuciones

    # Funcion para ejecutar el algoritmo
    def ejecutar(self, config_algorithm: BasePSO) -> None:

        # Si no hay problema definido, no hacemos nada
        if config_algorithm.datos_problema is None:

            # Mensaje de error
            if self.verbose:
                print('No hay problema definido')
            return

        # Lista que almacena las evaluaciones y generaciones de cada iteracion
        num_evaluaciones_iteracion: list = []
        num_generaciones_iteracion: list = []

        # Lista temporal para ir acumulando las poblaciones de cada nicho
        poblacion_acumulada: list = []
        
        # Bucle principal de ejecuciones iterativas
        for ejecucion in range(self.num_ejecuciones):

            # Mensaje indicativo
            if self.verbose:
                print(f'Running {ejecucion + 1}/{self.num_ejecuciones} examples:')

            # Hacemos la evalucion de un PSO
            super().ejecutar(config_algorithm)

            # Guardamos las evaluaciones y generaciones
            num_evaluaciones_iteracion.append(self.num_evaluaciones)
            num_generaciones_iteracion.append(self.num_iteraciones)

            # Guardamos la poblacion final
            poblacion_acumulada.extend(self.fitness_poblacion_final)

        # Sacamos la media de las iteraciones y evaluaciones de las n pasadas
        self.num_evaluaciones = sum(num_evaluaciones_iteracion)
        self.num_iteraciones = sum(num_generaciones_iteracion)

        # Asignamos todas las poblaciones finales
        self.fitness_poblacion_final = poblacion_acumulada
            
        # Mostramos las mejores soluciones
        for i, solucion in enumerate(self.mejores_soluciones):
            if self.verbose:
                print(f'Mejor resultado ({i}): {solucion.fitness}')