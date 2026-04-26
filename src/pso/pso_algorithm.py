# Librerias a usar
import random  # Modulo para la generacion de numeros pseudoaleatorios

from inspyred import swarm, ec  # Modulos del motor de inteligencia de enjambre e Inspyred
from src.common.base_algorithm import BaseAlgorithm  # Clase que vamos a heredar
from src.pso.base_pso import BasePSO  # Clase que sera el problema con las configuraciones necesarias


# Definimos la clase concreta para el algoritmo PSO, heredando de la clase abstracta
class PSOAlgorithm(BaseAlgorithm):
    
    # Constructor que inicializa los parametros del PSO y de la clase padre
    def __init__(self,
                    tamano_poblacion: int, max_evaluaciones: int,
                    inercia: float=0.7, cognitivo: float=1.5,
                    social: float=1.5, tamano_vecindario: int=5,
                    semilla: int=42
                ):
        
        # Invocamos al constructor de la clase padre para establecer las listas y variables base
        super().__init__(tamano_poblacion, max_evaluaciones)
        
        # Asignamos los parametros fisicos exclusivos del comportamiento del PSO
        self.inercia: float = inercia
        self.cognitivo: float = cognitivo
        self.social: float = social
        self.tamano_vecindario: int = tamano_vecindario
        
        # Inicializamos el motor aleatorio con la semilla para garantizar la repetibilidad
        self.generador_random: random.Random = random.Random(semilla)

    # Funcion para ejecutar el algoritmo
    def ejecutar(self, config_algorithm: BasePSO) -> None:
        
        # Informamos por consola del inicio de la configuracion
        print('CONFIG PSO')
        
        # Inicializamos el enjambre de particulas pasandole nuestro generador aleatorio
        pso_algorithm: swarm.PSO = swarm.PSO(self.generador_random)
        
        # Establecemos la condicion de parada estricta basada en el limite de evaluaciones
        pso_algorithm.terminator = ec.terminators.evaluation_termination
        
        # Asignamos los observers
        pso_algorithm.observer = [self._observer_fitness_diversidad, ec.observers.stats_observer]

        # Que cada clase haga las configuraciones que necesite
        config_algorithm.config_pso(pso_algorithm)
        
        # Informamos del inicio de las iteraciones
        print(f'Launching PSO ({self.max_evaluaciones} evals)')
        
        # Ejecutamos el motor pasando las reglas del problema y los parametros de vuelo
        poblacion_final: list = pso_algorithm.evolve(
            generator=config_algorithm.generator,
            evaluator=config_algorithm.evaluator,
            bounder=config_algorithm.bounder,
            maximize=config_algorithm.maximize,
            pop_size=self.tamano_poblacion,
            max_evaluations=self.max_evaluaciones,
            inertia=self.inercia,
            cognitive_rate=self.cognitivo,
            social_rate=self.social,
            neighborhood_size=self.tamano_vecindario
        )
        
        # Seleccionamos y guardamos la particula con el fitness mas optimo del resultado
        self.mejores_soluciones = self.mejores_soluciones = sorted(poblacion_final, key=lambda x: x.fitness)[:3]
        
        # Notificamos mejores fitness encontrados
        for i, solucion in enumerate(self.mejores_soluciones):
            print(f'Best solution #{i + 1}: {solucion.fitness}')