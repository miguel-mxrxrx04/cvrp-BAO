# Librerias a usar
import os  # Para uso de ficheros
import numpy as np  # Libreria para usar vectores en C
import concurrent.futures  # Modulo estandar para multiprocesamiento (Paralelismo real)

from inspyred.ec import Individual  # Clase que es un individuo (particula)
from src.pso.base_pso import BasePSO  # Clase base abstracta de las tecnicas multimodales
from src.common.problem import CVRPProblem  # Clase que define el problema a resolver
from src.pso.pso_algorithm import PSOAlgorithm  # Clase para crear el motor base del PSO
from src.common.clustering.join_routes import JoinRoutes  # Clase para juntar caminos y formar el viaje final sin pasarlos del limite de camiones
from src.common.clustering.enum_clustering_type import ClusteringTypes   # Enum para saber que tipo de clustering usar
from src.common.clustering.clustering_manager import ClusteringManager  # Clase que se encarga de dividir el problema
from src.pso.multimodal_techniques.sequential_pso import SequentialPSO  # Nuestra tecnica de Nichos Secuenciales
from src.common.clustering.clustering_problem import ClusterCVRPProblem  # Clase hija que vamos a instanciar


# Clase estatica encargada de ejecutar multiples instancias de PSO en paralelo dividiendo el mapa
class ClusteringPSOAlgorithm(PSOAlgorithm):

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
        tipo_clustering: ClusteringTypes=ClusteringTypes.NORMAL,
        max_clientes_cluster: int=40,
        truck_penalty: int=2,
        aplicar_reparacion: bool=True,
        do_parallel: bool=False,
        verbose: bool = False
    ):
        
        super().__init__(
            tamano_poblacion,
            max_evaluaciones,
            inercia, cognitivo,
            social,
            tamano_vecindario,
            semilla,
            verbose
        )

        # Numero de ejecuciones (lo haremos con nichos secuenciales al ser lo mas facil)
        self.num_ejecuciones: int = num_ejecuciones

        # Tipo de clustering que usaremos
        self.tipo_clustering: ClusteringManager = tipo_clustering.value

        # Numero de clientes por cluster
        self.num_clientes_cluster: int = max_clientes_cluster

        # Definimos si lo hacemos paralelo o no
        self.do_paralel: bool = do_parallel

        # Definimos si aplicamos reparacion de caminos o no (si nos excedemos en el numero de camiones)
        self.aplicar_reparacion: bool = aplicar_reparacion

        # Definimos la penalizacion por numero de camiones que nos pasemos
        self.truck_penalty: int = truck_penalty

    # Funcion para ejecutar el algoritmo
    def ejecutar(self, config_algorithm: BasePSO) -> None:

        # Si no hay problema definido, no hacemos nada
        if config_algorithm.datos_problema is None:

            # Mensaje de error
            if self.verbose:
                print('No hay problema definido')
            return

        # Obtenemos el problema principal
        problema_principal: ClusterCVRPProblem = config_algorithm.datos_problema

        # Definimos cuantos clientes (puntos de entrega) tendra cada cluster
        clientes_cluster: int = self.tipo_clustering.calcular_k_ideal(problema_principal.get_total_nodes(), self.num_clientes_cluster)

        # Lo dividimos en k sub-problemas
        sub_problemas: list = self.tipo_clustering.generar_sub_problemas(
            nodes=problema_principal.nodes,
            demands=problema_principal.demands,
            capacity=problema_principal.capacity,
            k_clusters=clientes_cluster,
            truck_penalty=self.truck_penalty
        )

        # Vemos como lo resolvemos
        if self.do_paralel:

            # Lo hacemos paralelo
            resultados_ejecucion: tuple = self._resolver_todo_en_paralelo(
                sub_problemas=sub_problemas,
                parametros_config_pso=config_algorithm.get_params_configuracion(),
                depot_id=problema_principal.depot_id
            )

        # Si no, lo hacemos secuencial
        else:

            # Hacemos en secuencial
            resultados_ejecucion: tuple = self._resolver_todo_en_secuencial(
                sub_problemas=sub_problemas,
                parametros_config_pso=config_algorithm.get_params_configuracion(),
                depot_id=problema_principal.depot_id
            )

        # Asignamos los valores a los atributos de la clase (historicos para ver evolucion de fitness y diversidad)
        self.mejores_soluciones = resultados_ejecucion[0]
        self.historico_fitness_mejor = resultados_ejecucion[1]
        self.historico_fitness_media = resultados_ejecucion[2]
        self.historico_diversidad = resultados_ejecucion[3]
        self.fitness_poblacion_final = resultados_ejecucion[4]

        # Aplicamos el join route si procede
        if self.aplicar_reparacion:

            # Lista que tendra las mejores soluciones con reparacion
            mejores_soluciones_reparadas: list = []
            
            # Bucle para aplicar la reparacion a las n mejores soluciones
            for i, solucion in enumerate(self.mejores_soluciones):
                
                # Aplicamos la reparacion
                ruta_reparada: list = JoinRoutes.consolidar_rutas(
                    rutas_pso=solucion,
                    demands=problema_principal.demands,
                    capacity=problema_principal.capacity,
                    limite_camiones=problema_principal.truck_limit,
                    depot_id=problema_principal.depot_id,
                    nodes=problema_principal.nodes
                )

                # Añadimos la solucion reparada junto con la sin reparar
                mejores_soluciones_reparadas.append(ruta_reparada)

        # Mostramos las mejores soluciones
        for i, solucion in enumerate(self.mejores_soluciones):

            # Mostramos la n mejor solucion
            self._mostrar_resultados(
                problema=problema_principal,
                limite_camiones=problema_principal.truck_limit,
                id_solucion=i,
                solucion=solucion
            )

            # Si hemos reparado, lo mostramos tambien
            if self.aplicar_reparacion:
                self._mostrar_resultados(
                    problema=problema_principal,
                    limite_camiones=problema_principal.truck_limit,
                    id_solucion=i,
                    solucion=mejores_soluciones_reparadas[i]
                )

                # Reemplazamos las mejores soluciones por las reparadas
                self.mejores_soluciones = mejores_soluciones_reparadas

    # Funcion que muestra el mejor fitness con los camiones usados
    def _mostrar_resultados(
        self,
        problema: ClusterCVRPProblem,
        limite_camiones: int,
        id_solucion: int,
        solucion: list,
    ) -> None:
        
        # Aplanamos la solucion para que el evaluador la entienda ---
        ruta_plana: list = []
        
        # Iteramos por cada viaje individual (camion)
        for viaje in solucion:
            
            # Si la ruta esta vacia, añadimos el primer viaje completo
            if not ruta_plana:
                ruta_plana.extend(viaje)
            
            # Si ya tiene datos, enganchamos el viaje omitiendo el primer deposito
            else:
                ruta_plana.extend(viaje[1:])

        # Calculamos el fitness usando la ruta ya aplanada
        fitness_solucion: float = problema.evaluate_route_distance(ruta_plana)

        # Extraemos el numero de camiones usados (la cantidad de sub-listas)
        camiones_usados: int = len(solucion)

        # Mostramos la solucion (fitness)
        if self.verbose:
            print(f'Mejor resultado ({id_solucion}): {fitness_solucion}')

        # Mostramos los camiones que usamos comparados con el limite teorico
        if self.verbose:
            print(f'Camiones usados: {camiones_usados}. Limite: {limite_camiones}')


    # Funcion aislada que ejecutara cada nucleo del procesador
    def _resolver_zona_completa(
        self,
        zona_id: int,
        problema_local: CVRPProblem,
        parametros_config_pso: dict,
        depot_id: int
    ) -> tuple:

        # Creamos el motor
        motor_local: PSOAlgorithm = PSOAlgorithm(
            tamano_poblacion=self.tamano_poblacion,
            max_evaluaciones=self.max_evaluaciones,
            inercia=self.inercia,
            cognitivo=self.cognitivo,
            social=self.social,
            tamano_vecindario=self.tamano_vecindario,
            verbose=self.verbose
        )
        
        # Lista que guarda las mejores rutas
        historial_mejores_rutas: list = []

        # Lista que guarda los caminos de los camiones en cada iteracion
        caminos_por_iteracion: list = []

        # Lista para acumular todas las poblaciones de las N ejecuciones
        fitness_total_zona: list = []

        # Lista que almacenara el fitness y diversidad de cada ejecucion
        total_mejor_fitness: list = []
        total_media_fitness: list = []
        total_diversidad: list = []
        
        # Bucle Secuencial: Las iteraciones dependen del tiempo, se hacen una tras otra en el mismo nucleo
        for _ in range(self.num_ejecuciones):
            
            # Instanciamos la tecnica Secuencial con el detector de Jaccard
            tecnica_secuencial: SequentialPSO = SequentialPSO(
                datos_problema=problema_local,
                umbral_similitud=parametros_config_pso['umbral_similitud'],
                penalizacion=parametros_config_pso['penalizacion'],
                num_soluciones_corregir=parametros_config_pso['num_soluciones_corregir']
            )
            
            # Le inyectamos la memoria historica de las rutas prohibidas (mejores)
            tecnica_secuencial.optimos_encontrados = historial_mejores_rutas.copy()
            
            # Limpiamos las particulas anteriores, pero el motor sigue vivo con su entropia
            motor_local.clear() 
            motor_local.ejecutar(tecnica_secuencial)
            
            # Extraemos y limpiamos la mejor particula de este vuelo
            mejor_particula: Individual = motor_local.mejores_soluciones[0]
            ruta_mejor_particula: list = tecnica_secuencial.get_ruta_particula(mejor_particula)
            
            # Guardamos la ruta limpia real para penalizarla en la siguiente iteracion
            historial_mejores_rutas.append(ruta_mejor_particula)

            # Guardamos los historicos fitness y diversidad
            total_mejor_fitness.append(motor_local.historico_fitness_mejor)
            total_media_fitness.append(motor_local.historico_fitness_media)
            total_diversidad.append(motor_local.historico_diversidad)
            fitness_total_zona.extend(motor_local.fitness_poblacion_final)
            
            # Obtenemos una lista con los caminos de los camiones
            caminos_solucion: list = []

            # El inicio es el nodo del deposito
            viaje_actual: list = [depot_id]

            # Viajamos por los nodos (empezamos en 1 porque es el deposito y ya lo almacenamos)
            for nodo in ruta_mejor_particula[1:]:

                # Lo añadimos al camino actuañ
                viaje_actual.append(nodo)

                # Vemos si estamos en un deposito
                if nodo == depot_id:

                    # Añadimos el nuevo camino (con la ida y retorno)
                    caminos_solucion.append(viaje_actual)

                    # Creamos el nuevo camino (de otro camion)
                    viaje_actual = [depot_id]
                    
            # Guardamos la bolsa de camiones de esta iteracion
            caminos_por_iteracion.append(caminos_solucion)
            
        # Sacamos la media de los historicos
        mean_total_mejor_fitness: list = np.mean(total_mejor_fitness, axis=0).tolist()
        mean_total_media_fitness: list = np.mean(total_media_fitness, axis=0).tolist()
        mean_total_diversidad: list = np.mean(total_diversidad, axis=0).tolist()
        
        # Retornamos la matriz de camiones (3 iteraciones x N camiones) y los historicos de fitness (mejor y medio) y diversidad
        return (
            caminos_por_iteracion,
            mean_total_mejor_fitness,
            mean_total_media_fitness,
            mean_total_diversidad,
            fitness_total_zona
        )

    # Funcion que resuelve el problema de manera secuencial
    def _resolver_todo_en_secuencial(
        self,
        sub_problemas: list,
        parametros_config_pso: dict,
        depot_id: int,
    ) -> tuple:
        
        # Preparamos N listas vacias por caminos encontrados
        caminos_totales: list = [[] for _ in range(self.num_ejecuciones)]

        # Lista con el fitness de la poblacion (global, para graficas estadisticas)
        fitness_poblacion_global: list = []

        # Lista con los historiales de las zonas (fitness)
        historial_mejor_fitness_zonas: list = []
        historial_media_fitness_zonas: list = []
        historial_diversidad_zonas: list = []

        # Iteramos por cada sub-problema (zona) procesandolas una tras otra
        for i, problema in enumerate(sub_problemas):

            # Resolvemos la zona actual en el hilo principal
            caminos_zonas, best_fitness_zona, mean_fitness_zona, diversidad_zona, fitness_poblacion_zona = self._resolver_zona_completa(
                zona_id=i,
                problema_local=problema,
                parametros_config_pso=parametros_config_pso,
                depot_id=depot_id
            )

            # Guardamos el historial de esta zona para hacer la media luego
            historial_mejor_fitness_zonas.append(best_fitness_zona)
            historial_media_fitness_zonas.append(mean_fitness_zona)
            historial_diversidad_zonas.append(diversidad_zona)
            fitness_poblacion_global.extend(fitness_poblacion_zona)
            
            # Repartimos los camiones devueltos en su lista global correspondiente
            for it in range(self.num_ejecuciones):
                caminos_totales[it].extend(caminos_zonas[it])

        # Hacemos la media vertical del fitness y diversidad
        historial_mean_best_fitness: list = np.mean(historial_mejor_fitness_zonas, axis=0).tolist()
        historial_mean_media_fitness: list = np.mean(historial_media_fitness_zonas, axis=0).tolist()
        historial_mean_diversidad: list = np.mean(historial_diversidad_zonas, axis=0).tolist()
                
        # Devolvemos las bolsas ensambladas listas para JoinRoutes
        return (
            caminos_totales,
            historial_mean_best_fitness,
            historial_mean_media_fitness,
            historial_mean_diversidad,
            fitness_poblacion_global
        )

    # Funcion que orquesta el paralelismo distribuyendo las zonas geograficas
    def _resolver_todo_en_paralelo(
        self,
        sub_problemas: list,
        parametros_config_pso: dict,
        depot_id: int,
    ) -> tuple:
        
        # Preparamos N listas vacias por caminos encontrados
        caminos_totales: list = [[] for _ in range(self.num_ejecuciones)]

        # Lista con el fitness de la poblacion (global, para graficas estadisticas)
        fitness_poblacion_global: list = []

        # Dejamos 2 hilos libres para que el sistema operativo y VSC no se congelen
        n_hilos_seguros: int = max(1, os.cpu_count() - 3)
        
        # Levantamos el Pool de procesos
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_hilos_seguros) as ejecutor:
            
            # Preparamos las promesas (futuros) lanzando todas las zonas a la vez
            futuros: list = [
                ejecutor.submit(
                    self._resolver_zona_completa,
                    i,
                    problema,
                    parametros_config_pso, 
                    depot_id,
                )
                for i, problema in enumerate(sub_problemas)
            ]

            # Lista con los historiales de las zonas (fitness)
            historial_mejor_fitness_zonas: list = []
            historial_media_fitness_zonas: list = []
            historial_diversidad_zonas: list = []
            
            # A medida que los nucleos van terminando todo su trabajo, recogemos los resultados
            for futuro in concurrent.futures.as_completed(futuros):
                
                # Extraemos los resultados de la solucion de la zona (iterado n veces)
                caminos_zonas, best_fitness_zona, mean_fitness_zona, diversidad_zona, fitness_poblacion_zona = futuro.result()

                # Guardamos el historial de esta zona para hacer la media luego
                historial_mejor_fitness_zonas.append(best_fitness_zona)
                historial_media_fitness_zonas.append(mean_fitness_zona)
                historial_diversidad_zonas.append(diversidad_zona)
                fitness_poblacion_global.extend(fitness_poblacion_zona)
                
                # Repartimos los camiones devueltos en su lista global correspondiente
                for it in range(self.num_ejecuciones):
                    caminos_totales[it].extend(caminos_zonas[it])

        # Hacemos la media vertical del fitness y diversidad
        historial_mean_best_fitness: list = np.mean(historial_mejor_fitness_zonas, axis=0).tolist()
        historial_mean_media_fitness: list = np.mean(historial_media_fitness_zonas, axis=0).tolist()
        historial_mean_diversidad: list = np.mean(historial_diversidad_zonas, axis=0).tolist()
                
        # Devolvemos las bolsas ensambladas listas para JoinRoutes
        return (
            caminos_totales,
            historial_mean_best_fitness,
            historial_mean_media_fitness,
            historial_mean_diversidad,
            fitness_poblacion_global
        )