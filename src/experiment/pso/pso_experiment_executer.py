# Librerias a usar
import os  # Modulo estandar para interactuar con el sistema de archivos
import time  # Modulo estandar para medir tiempos de ejecucion del codigo
import copy  # Para sacar copias
import shutil  # Libreria estandar para borrado de directorios con contenido
import random  # Clase para sacar numeros aleatorios
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

        # Leemos todos los archivos ubicados en el directorio especificado
        archivos_carpeta: list = os.listdir(self.data_path)

        # Filtramos y almacenamos unicamente los nombres que terminan en la extension vrp
        self.instancias_vrp: list = [archivo for archivo in archivos_carpeta if archivo.endswith('.vrp')]

    # Funcion que asigna hiperparametros segun la magnitud del problema
    def _calculate_params(self, n_clientes: int) -> dict:
        
        # Declaramos un diccionario base con los parametros estaticos
        parametros: dict = {
            'inercia': 0.7,
            'cognitivo': 1.5,
            'social': 1.5,
            'tamano_vecindario': 5
        }

        # Escalado automatico evaluando la cantidad de nodos
        if n_clientes <= 50:

            # Asignamos el limite de evaluaciones para mapas pequeños o medianos
            parametros['max_evaluaciones'] = 3000
            parametros['tamano_poblacion'] = 20

        elif n_clientes <= 200:
            
            # Asignamos el limite de evaluaciones para mapas pequeños o medianos
            parametros['max_evaluaciones'] = 10000
            parametros['tamano_poblacion'] = 50
            
        else:
            
            # Asignamos un limite mayor de evaluaciones para mapas complejos
            parametros['max_evaluaciones'] = 30000
            parametros['tamano_poblacion'] = 100
            
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

        # Extraemos el mejor fitness alcanzado (el ultimo valor guardado en el historico)
        mejor_fitness_corrida: float = algoritmo_ejecutado.historico_fitness_mejor[-1]

        # Estructuramos y devolvemos el diccionario con las metricas solicitadas
        return {
            'run': run_id,
            'fitness': mejor_fitness_corrida,
            'n_evaluations': algoritmo_ejecutado.max_evaluaciones,
            'experiment_id': nombre_problema.split('.')[0],
            'tiempo': tiempo_ejecucion
        }

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

        # Evaluamos si el directorio de exportacion ya existe
        if os.path.isdir(experiment_folder):
            
            # Si el usuario permite sobreescribir, borramos la carpeta entera
            if overwrite:
                shutil.rmtree(experiment_folder)
                
            # Si no permite sobreescribir, lanzamos una excepcion para proteger los datos
            else:
                raise ValueError(f'La carpeta {experiment_folder} ya existe. Usa overwrite=True para sobreescribirla.')

        # Generamos la jerarquia de carpetas limpia
        os.makedirs(experiment_folder)

        # Entramos en un barrido iterativo procesando cada documento (.vrp) localizado
        for instancia in self.instancias_vrp:
            
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
            
            # Notificamos el exito materializando la pervivencia de los registros
            print(f'Salida generada exitosamente en: {ruta_exportacion}')

    # Funcion auxiliar para leer rapidamente la cantidad de clientes de un archivo
    def _obtener_num_clientes(self, nombre_problema: str) -> int:
        
        # Reconstruimos la ruta del archivo
        ruta_problema: str = os.path.join(self.data_path, nombre_problema)
        
        # Usamos el parser solo para extraer los nodos
        parser_temporal: Parser = Parser(ruta_problema)
        nodos, _, _ = parser_temporal.parse()
        
        # Retornamos la cantidad de clientes (total de nodos menos el deposito)
        return len(nodos) - 1