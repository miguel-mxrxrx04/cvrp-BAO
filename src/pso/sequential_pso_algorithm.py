# Librerias a usar
import random  # Modulo para la generacion de numeros pseudoaleatorios

from inspyred import swarm, ec  # Modulos del motor de inteligencia de enjambre e Inspyred
from src.pso.pso_algorithm import PSOAlgorithm  # Heredaremos de este cambiando solo la ejecucion
from src.pso.base_pso import BasePSO  # Clase para tipado

class SequentialPSOAlgorithm(PSOAlgorithm):
    
    # Añadimos un parametro nuevo: el numero de veces que lanzaremos el enjambre
    def __init__(self,
                    tamano_poblacion: int, max_evaluaciones: int,
                    inercia: float=0.7, cognitivo: float=1.5,
                    social: float=1.5, tamano_vecindario: int=5,
                    semilla: int=42
                ):
        
        super().__init__(
            tamano_poblacion, max_evaluaciones,
            inercia, cognitivo, social
            , tamano_vecindario, semilla
        )

    # Funcion para ejecutar el algoritmo
    def ejecutar(self, config_algorithm: BasePSO, num_ejecuciones: int=3) -> None:
        
        # Bucle principal de ejecuciones iterativas
        for ejecucion in range(num_ejecuciones):

            # Mensaje indicativo
            print(f'Running {ejecucion + 1}/{num_ejecuciones} examples:')

            # Hacemos la evalucion de un PSO
            super().ejecutar(config_algorithm)
            
        # Mostramos las mejores soluciones
        for i, solucion in enumerate(self.mejores_soluciones):
            print(f'Mejor resultado ({i}): {solucion.fitness}')