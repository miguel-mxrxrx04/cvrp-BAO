# Librerias a usar
import os  # Modulo estandar para interactuar con el sistema de archivos
import re  # Para expresiones regulares
import time  # Modulo estandar para medir tiempos de ejecucion del codigo
import copy  # Para sacar copias
import shutil  # Libreria estandar para borrado de directorios con contenido
import random  # Clase para sacar numeros aleatorios
import numpy as np  # Libreria para el manejo de arrays eficiente y operaciones matematicas
import pandas as pd  # Libreria externa para manipulacion y creacion de dataframes
import concurrent.futures  # Para hacerlo paralelo

from pathlib import Path  # Modulo avanzado para interactuar con el sistema de archivos
from src.common.parser import Parser  # Clase para parsear los archivos de las instancias vrp
from src.pso.base_pso import BasePSO  # Clase base abstracta de las tecnicas multimodales
from src.common.problem import CVRPProblem  # Clase que modela las reglas fisicas del enrutamiento
from src.common.diversity import DiversityHandler  # Clase para el calculo de similitudes estructurales
from src.common.base_algorithm import BaseAlgorithm  # Clase que se encarga de ejecutar el problema
from src.pso.clustering_pso_algorithm import ClusteringPSOAlgorithm  # Clase para ejecutar el enjambre concurrente
from src.common.clustering.clustering_manager import ClusteringManager  # Clase para calcular metricas de particionado


# Clase encargada de automatizar la bateria de experimentos estadisticos
class PSOExperimentExecuter:

    # Constructor de la clase que inicializa la ruta de los datos
    def __init__(self, data_path: str):

        # Guardamos la ruta proporcionada como atributo de instancia
        ruta_script = Path(__file__).resolve()
        self.data_path: str = ruta_script.parents[3] / data_path

        # Obtenemos los problemas divididos
        problemas_divididos: tuple = self._dividir_problemas()

        # Asignamos a los atributos correspondientes
        self.problemas_pequenos: set = problemas_divididos[0]
        self.problemas_medianos: set = problemas_divididos[1]
        self.problemas_grandes: set = problemas_divididos[2]

        # Numero de problemas disponibles
        self.num_problemas: int = len(self.problemas_pequenos) + len(self.problemas_medianos) + len(self.problemas_grandes)

    # Funcion que divide los problemas en pequeños, medianos y grandes
    def _dividir_problemas(self) -> tuple:

        # Leemos todos los archivos ubicados en el directorio especificado
        archivos_carpeta: list = os.listdir(self.data_path)

        # Filtramos y almacenamos unicamente los nombres que terminan en la extension vrp
        instancias_vrp: list = [archivo for archivo in archivos_carpeta if archivo.endswith('.vrp')]

        # Conjuntos para almacenar los problemas clasificados por complejidad (ahora como conjuntos)
        problemas_pequenos: set = set()
        problemas_medianos: set = set()
        problemas_grandes: set = set()

        # Instanciamos el patron para extraer el numero de clientes
        patron = re.compile(r'n(\d+)-k')

        # Clasificamos cada instancia leida en su atributo correspondiente
        for archivo in instancias_vrp:

            # Obtenemos el problema
            match = patron.search(archivo)
    
            # Vemos si lo encontramos
            if match:

                # Sacamos el numero de nodos
                nodos: int = int(match.group(1))

                # Si son pocos
                if nodos < 300:

                    # Guardamos en lista de pequeños
                    problemas_pequenos.add(archivo)

                # Si son medianos
                elif nodos < 750:

                    # Guardamos en lista de medianos
                    problemas_medianos.add(archivo)

                # Si son grandes
                else:

                    # Guaradamos en lista de grandes
                    problemas_grandes.add(archivo)

        # Devolvemos los conjuntos
        return (
            problemas_pequenos,
            problemas_medianos,
            problemas_grandes
        )

    # Funcion que asigna hiperparametros segun la magnitud del problema
    def _calculate_params(self, n_clientes: int) -> dict:
        
        # Declaramos un diccionario base con los parametros estaticos
        parametros: dict = {}

        # Escalado automatico evaluando la cantidad de nodos (este por si es de un cluster)
        if n_clientes <= 50:

            # Asignamos el limite de evaluaciones para mapas pequeños o medianos
            parametros['max_evaluaciones'] = 3000
            parametros['tamano_poblacion'] = 20

        # Si es un mapa pequeño
        elif n_clientes <= 300:
            
            # Asignamos el limite de evaluaciones para mapas pequeños o medianos
            parametros['max_evaluaciones'] = 10000
            parametros['tamano_poblacion'] = 50

        # Si es un mapa pequeño
        elif n_clientes <= 750:
            
            # Asignamos el limite de evaluaciones para mapas pequeños o medianos
            parametros['max_evaluaciones'] = 30000
            parametros['tamano_poblacion'] = 100
        
        # SI es un mapa gigante
        else:
            
            # Asignamos un limite mayor de evaluaciones para mapas complejos
            parametros['max_evaluaciones'] = 45000
            parametros['tamano_poblacion'] = 150
            
        # Devolvemos el diccionario perfectamente configurado
        return parametros

    # Funcion que procesa una unica evaluacion algoritmica completa
    def run_single_experiment(
        self,
        nombre_problema: str,
        pso_algorithm: BaseAlgorithm,
        pso_config: BasePSO,
    ) -> BaseAlgorithm:

        # Ruta del problema
        ruta_problema: str = os.path.join(self.data_path, nombre_problema)

        # Obtenemos el limite de camiones
        limite_camiones: int = int(nombre_problema.split('k')[1].split('.')[0])

        # Generamos la instancia del parseador pasandole la direccion fisica del archivo
        parser: Parser = Parser(ruta_problema)

        # Ejecutamos la lectura devolviendo los diccionarios con la topologia del terreno
        nodos, demandas, capacidad = parser.parse()

        # Montamos el entorno fisico del problema mediante la clase gestora
        datos_problema: CVRPProblem = CVRPProblem(nodos, demandas, capacidad, limite_camiones)

        # Le damos el problema al config
        pso_config.asignar_problema(datos_problema)

        # Ejecutamos el PSO
        pso_algorithm.ejecutar(pso_config)

        # Entregamos el algoritmo ya ejecutado
        return pso_algorithm

    # Funcion que ejecuta una configuracion estatica repetidas veces
    def run_repeated_experiment(
        self,
        nombre_problema: str,
        pso_algorithm: BaseAlgorithm,
        pso_config: BasePSO,
        n_repeat: int=35,
        do_parallel: bool=False
    ) -> pd.DataFrame:

        # Evaluamos la bandera de paralelismo para enrutar el flujo
        if do_parallel:

            # Retornamos el flujo paralelo
            return self._run_repeated_parallel(
                nombre_problema=nombre_problema,
                pso_algorithm=pso_algorithm,
                pso_config=pso_config,
                n_repeat=n_repeat
            )

        else:

            # Retornamos el flujo secuencial por defecto
            return self._run_repeated_sequential(
                nombre_problema=nombre_problema,
                pso_algorithm=pso_algorithm,
                pso_config=pso_config,
                n_repeat=n_repeat
            )

    # Funcion auxiliar para aislar la ejecucion de una sola repeticion (usada por secuencial y paralelo)
    def _run_single_repetition_worker(
        self,
        run_id: int,
        nombre_problema: str,
        pso_algorithm: BaseAlgorithm,
        pso_config: BasePSO
    ) -> dict:

        # Reportamos por consola el estado de progresion
        print(f'Procesando experimento: {nombre_problema} - run {run_id + 1}')

        # Limpiamos el historico del objeto para no arrastrar datos de la iteracion anterior
        pso_algorithm.clear()

        # Generamos una nueva semilla (evitamos repeticiones)
        nueva_semilla: int = random.SystemRandom().randint(0, 9999999) + run_id
        pso_algorithm.generador_random.seed(nueva_semilla)

        # Registramos el reloj del sistema para computar duraciones absolutas
        inicio_tiempo: float = time.time()

        # Ejecutamos el experimento usando la instancia reseteada
        algoritmo_ejecutado: BaseAlgorithm = self.run_single_experiment(
            nombre_problema,
            pso_algorithm,
            pso_config
        )

        # Calculamos el tiempo total transcurrido
        tiempo_ejecucion: float = time.time() - inicio_tiempo

        # Sacamos datos para el csv
        datos_csv: tuple = self._sacar_datos_csv(algoritmo_ejecutado, pso_config)

        # Estructuramos y devolvemos el diccionario con las metricas solicitadas
        return {
            'run': run_id,
            'mean_fitness': datos_csv[0],
            'std_fitness': datos_csv[1],
            'mean_jaccard_distance': datos_csv[2],
            'n_evaluations': algoritmo_ejecutado.num_evaluaciones,
            'n_generations': algoritmo_ejecutado.num_iteraciones,
            'experiment_id': nombre_problema.split('.')[0],
            'tiempo': tiempo_ejecucion
        }

    # Funcion principal que orquesta la extraccion de datos para el CSV
    def _sacar_datos_csv(self, pso_algorithm: BaseAlgorithm, pso_config: BasePSO) -> tuple:
        
        # Si es una lista, estamos ante el Clustering. Si no, es el PSO Base (Individual)
        es_clustering: bool = isinstance(pso_algorithm.mejores_soluciones[0], list)

        # Delegamos la obtencion de fenotipos y fitness segun el tipo de algoritmo
        if es_clustering:
            rutas, fits = self._procesar_fenotipos_clustering(pso_algorithm.mejores_soluciones, pso_config)
        else:
            rutas, fits = self._procesar_fenotipos_pso(pso_algorithm.mejores_soluciones, pso_config)

        # Delegamos el calculo de la diversidad estructural
        mean_jaccard: float = self._calcular_diversidad_jaccard(rutas)

        # Retornamos la tupla final (media_fitness, std_fitness, diversidad_media)
        return (float(np.mean(fits)), float(np.std(fits)), mean_jaccard)

    # Funcion especializada en decodificar y aplanar las soluciones de Clustering
    def _procesar_fenotipos_clustering(self, soluciones: list, pso_config: BasePSO) -> tuple:
        
        # Aplanamos la estructura de camiones (Ejecucion -> Camion -> Nodo)
        rutas_planas: list = [sum([v if i == 0 else v[1:] for i, v in enumerate(sol)], []) for sol in soluciones]
        
        # Evaluamos el coste real de cada ruta aplanada
        fitness_list: list = [pso_config.datos_problema.evaluate_route_distance(r) for r in rutas_planas]
        
        # Devolvemos las rutas y su fitness
        return rutas_planas, fitness_list

    # Funcion especializada en decodificar los individuos del PSO Base
    def _procesar_fenotipos_pso(self, individuos: list, pso_config: BasePSO) -> tuple:
        
        # Obtenemos el fenotipo real (ruta de nodos) mediante el decodificador de la configuracion
        rutas_fenotipo: list = [pso_config.get_ruta_particula(ind) for ind in individuos]
        
        # Extraemos el fitness ya calculado en la particula
        fitness_list: list = [ind.fitness for ind in individuos]
        
        # Devolvemos las rutas y su fitness
        return rutas_fenotipo, fitness_list

    # Funcion encargada de calcular la media de Jaccard entre todas las soluciones
    def _calcular_diversidad_jaccard(self, rutas: list) -> float:
        
        # Generamos la lista de distancias comparando todos los pares posibles (sin repeticion)
        n: int = len(rutas)
        distancias: list = [
            DiversityHandler.calculate_jaccard_distance(rutas[i], rutas[j]) 
            for i in range(n) for j in range(i + 1, n)
        ]
        
        # Retornamos el promedio (0.0 si no hay comparaciones disponibles)
        return float(np.mean(distancias)) if distancias else 0.0

    # Funcion que ejecuta la bateria de experimentos de forma secuencial (clasica)
    def _run_repeated_sequential(
        self,
        nombre_problema: str,
        pso_algorithm: BaseAlgorithm,
        pso_config: BasePSO,
        n_repeat: int
    ) -> pd.DataFrame:

        # Generamos una lista vacia para acumular los registros del muestreo estadistico
        resultados_estadisticos: list = []

        # Disparamos el ciclo con la cantidad de repeticiones impuesta
        for run in range(n_repeat):

            # Invocamos al trabajador aislado
            resultado_fila: dict = self._run_single_repetition_worker(
                run_id=run,
                nombre_problema=nombre_problema,
                pso_algorithm=pso_algorithm,
                pso_config=pso_config
            )

            # Agregamos el registro tabular a la lista principal
            resultados_estadisticos.append(resultado_fila)

        # Convertimos la lista de diccionarios en un dataframe y lo retornamos
        return pd.DataFrame(resultados_estadisticos)

    # Funcion que ejecuta la bateria de experimentos en paralelo (Macro-Paralelismo)
    def _run_repeated_parallel(
        self,
        nombre_problema: str,
        pso_algorithm: BaseAlgorithm,
        pso_config: BasePSO,
        n_repeat: int
    ) -> pd.DataFrame:

        # Generamos una lista vacia para acumular los registros
        resultados_estadisticos: list = []

        # Dejamos 3 hilos libres para que el sistema operativo y entorno no se congelen
        n_hilos_seguros: int = max(1, os.cpu_count() - 3)

        # Levantamos el Pool de procesos
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_hilos_seguros) as ejecutor:

            futuros: list = []

            # Repartimos las repeticiones como tareas independientes
            for run in range(n_repeat):

                # Clonamos los objetos (¡Vital! para evitar colisiones en memoria entre nucleos)
                alg_copy = copy.deepcopy(pso_algorithm)
                config_copy = copy.deepcopy(pso_config)

                # Asignamos la tarea al pool
                futuro = ejecutor.submit(
                    self._run_single_repetition_worker,
                    run,
                    nombre_problema,
                    alg_copy,
                    config_copy
                )
                futuros.append(futuro)

            # Recolectamos los resultados a medida que terminan
            for futuro in concurrent.futures.as_completed(futuros):
                try:
                    resultado_fila: dict = futuro.result()
                    resultados_estadisticos.append(resultado_fila)
                except Exception as e:
                    print(f"Error critico en una ejecucion paralela: {e}")

        # Pasamos a dataframe, ordenamos por 'run' para mantener el orden original y reseteamos el indice
        df_resultados = pd.DataFrame(resultados_estadisticos)
        df_resultados = df_resultados.sort_values(by=['run']).reset_index(drop=True)

        # Devolvemos los resultados
        return df_resultados

    # Funcion que evalua de forma global todas las instancias almacenadas
    def run_all_experiments(
        self,
        experiment_folder: str,
        pso_algorithm: BaseAlgorithm,
        pso_config: BasePSO,
        num_clientes_cluster: int=0,
        n_repeat: int=35,
        do_paralel: bool=False,
        overwrite: bool=False
    ) -> None:

        # Conjunto para almacenar que CSVs ya existen
        archivos_procesados: set = set()

        # Evaluamos si el directorio de exportacion ya existe
        if os.path.isdir(experiment_folder):
            
            # Vemos si quiere sobreescribir
            if overwrite:

                # Borramos la carpeta entera
                shutil.rmtree(experiment_folder)
                
            # Si no permite sobreescribir, lanzamos prueba con los que no hicimos csv
            else:
                
                # Sacamos los problemas resueltos
                self._obtener_problemas_resueltos(archivos_procesados, experiment_folder)

        # Juntamos todo lo que vamos a procesar
        instancias_vrp: list = self._obtener_problemas_procesar(archivos_procesados)

        # Generamos la jerarquia de carpetas limpia
        os.makedirs(experiment_folder, exist_ok=True)

        # Si no hay problemas para procesar, terminamos
        if not instancias_vrp:

            # Mensaje
            print('No hay problemas para procesar, estan todos.')
            return

        # Entramos en un barrido iterativo procesando cada documento (.vrp) localizado
        for instancia in instancias_vrp:
            
            # Sondenamos el tamano del mapa real para ajustar el esfuerzo computacional
            total_clientes: int = self._obtener_num_clientes(instancia) if num_clientes_cluster == 0 else num_clientes_cluster
            
            # Calculamos dinamicamente los hiperparametros escalados
            parametros_escala: dict = self._calculate_params(total_clientes)
            
            # Inyectamos los parametros de forma dinamica al motor PSO antes de ejecutarlo
            pso_algorithm.max_evaluaciones = parametros_escala['max_evaluaciones']
            pso_algorithm.tamano_poblacion = parametros_escala['tamano_poblacion']
            
            # Emitimos una notificacion anunciando la apertura de un bloque masivo
            print(f'Iniciando pruebas para: {instancia} ({total_clientes} clientes) | Evaluaciones: {pso_algorithm.max_evaluaciones}')

            # Ejecutamos el lote de N repeticiones para este mapa
            df_resultados: pd.DataFrame = self.run_repeated_experiment(
                nombre_problema=instancia,
                pso_algorithm=pso_algorithm,
                pso_config=pso_config,
                n_repeat=n_repeat,
                do_parallel=do_paralel
            )

            # Purificamos el nombre original retirando extensiones de formato
            nombre_limpio: str = instancia.split('.')[0]
            
            # Añadimos la columna identificadora para cuando juntemos los CSVs despues
            df_resultados['experiment_id'] = nombre_limpio

            # Combinamos rutas y marcadores logrando una ruta de escritura segura
            ruta_exportacion: str = os.path.join(experiment_folder, f'experiment_{nombre_limpio}.csv')

            # Volcamos la matriz de datos omitiendo el indice enumerativo nativo
            df_resultados.to_csv(ruta_exportacion, index=False)

    # Funcion que saca los problemas ya procesados
    def _obtener_problemas_resueltos(self, archivos_procesados: set, experiment_folder: str) -> None:

        # Patron para capturar todo lo que hay entre 'experiment_' y '.csv'
        patron_csv = re.compile(r'experiment_(.*)\.csv')

        # Bucle para obtener los problemas ya resueltos
        for archivo in os.listdir(experiment_folder):

            # Hacemos la RE para sacar el nombre del archivo
            match_csv = patron_csv.match(archivo)
                    
            # Reconstruimos el nombre base con su extension original
            nombre_base: str = f'{match_csv.group(1)}.vrp'
            archivos_procesados.add(nombre_base)

    # Funcion para obtener los problemas a procesar
    def _obtener_problemas_procesar(self, archivos_procesados: set) -> list:

        # Sacamos los archivos ya existentes
        problemas_pequeños: list = list(self.problemas_pequenos - archivos_procesados)
        problemas_medianos: list = list(self.problemas_medianos - archivos_procesados)
        problemas_grandes: list = list(self.problemas_grandes - archivos_procesados)

        # Sacamos una muestra de cada uno aleatoriamente (mas muestras para los pequeños, menos para las grandes)
        procesar_pequeños = set(random.sample(problemas_pequeños, min(14, len(problemas_pequeños))))
        procesar_medianos = set(random.sample(problemas_medianos, min(5, len(problemas_medianos))))
        procesar_grandes = set(random.sample(problemas_grandes, min(2, len(problemas_grandes))))

        # Devolvemos los problemas a procesar
        return list(procesar_pequeños | procesar_medianos | procesar_grandes)

    # Funcion auxiliar para leer rapidamente la cantidad de clientes de un archivo
    def _obtener_num_clientes(self, nombre_problema: str) -> int:
        
        # Reconstruimos la ruta del archivo
        ruta_problema: str = os.path.join(self.data_path, nombre_problema)
        
        # Usamos el parser solo para extraer los nodos
        parser_temporal: Parser = Parser(ruta_problema)
        nodos, _, _ = parser_temporal.parse()
        
        # Retornamos la cantidad de clientes (total de nodos menos el deposito)
        return len(nodos) - 1