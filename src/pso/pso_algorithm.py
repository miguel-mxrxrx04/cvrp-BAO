# Librerias a usar
import random  # Modulo para la generacion de numeros pseudoaleatorios

from inspyred import swarm, ec  # Modulos del motor de inteligencia de enjambre e Inspyred
from src.common.base_algorithm import BaseAlgorithm  # Clase que vamos a heredar
from src.pso.base_pso import BasePSO  # Clase que sera el problema con las configuraciones necesarias


# Definimos la clase concreta para el algoritmo PSO, heredando de la clase abstracta
class PSOAlgorithm(BaseAlgorithm):
    
    # Constructor que inicializa los parametros del PSO y de la clase padre
    def __init__(
        self,
        tamano_poblacion: int,
        max_evaluaciones: int,
        inercia: float=0.7,
        cognitivo: float=1.5,
        social: float=1.5,
        tamano_vecindario: int=5,
        semilla: int=42,
        verbose: bool=False
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

        # Bool para saber si usamos el observer o no
        self.verbose: bool = verbose

    # Funcion para ejecutar el algoritmo
    def ejecutar(self, config_algorithm: BasePSO) -> None:

        # Si no hay problema definido, no hacemos nada
        if config_algorithm.datos_problema is None:

            # Mensaje de error
            if self.verbose:
                print('No hay problema definido')
            return
        
        # Informamos por consola del inicio de la configuracion
        if self.verbose:
            print('CONFIG PSO')
        
        # Inicializamos el enjambre de particulas pasandole nuestro generador aleatorio
        pso_algorithm: swarm.PSO = swarm.PSO(self.generador_random)
        
        # Establecemos la condicion de parada estricta basada en el limite de evaluaciones
        pso_algorithm.terminator = [
            ec.terminators.evaluation_termination,
            ec.terminators.no_improvement_termination
        ]
        
        # Asignamos los observers
        if self.verbose:
            pso_algorithm.observer = [self._observer_fitness_diversidad, ec.observers.stats_observer]
        else:
            pso_algorithm.observer = [self._observer_fitness_diversidad]

        # Que cada clase haga las configuraciones que necesite
        config_algorithm.config_pso(pso_algorithm)
        
        # Informamos del inicio de las iteraciones
        if self.verbose:
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

        # Actualizamos el numero de evaluaciones y generaciones reales
        self.num_evaluaciones = pso_algorithm.num_evaluations
        self.num_iteraciones = pso_algorithm.num_generations
        
        # Seleccionamos y guardamos la particula con el fitness mas optimo del resultado
        self.mejores_soluciones = sorted(poblacion_final, key=lambda x: x.fitness)[:3]

        # Guardamos el fitness de las particulas para el Histograma y Box-plot
        self.fitness_poblacion_final = [ind.fitness for ind in poblacion_final]
        
        # Notificamos mejores fitness encontrados
        for i, solucion in enumerate(self.mejores_soluciones):
            if self.verbose:
                print(f'Best solution #{i + 1}: {solucion.fitness}')