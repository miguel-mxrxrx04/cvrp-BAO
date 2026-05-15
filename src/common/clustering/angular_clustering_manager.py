# Librerias a usar
import math  # Modulo matematico para calculos trigonometricos y absolutos
import numpy as np  # Manejo eficiente de vectores numericos

from typing import Optional  # Para el tipado
from sklearn.cluster import KMeans  # Modulo para aplicar el algoritmo K-Means
from src.common.clustering.clustering_manager import ClusteringManager  # Clase padre a heredar

# Clase hija que hereda la logica de agrupacion pero cambia el motor matematico a barrido polar (Sweep Algorithm) de 5 Dimensiones
class AngularClusteringManager(ClusteringManager):

    # Funcion principal que orquesta la division llamando a los metodos privados
    @staticmethod
    def generar_sub_problemas(nodes: dict, demands: dict, capacity: int, k_clusters: int, truck_penalty: float=2) -> list:
        
        # Extraemos el ID del deposito (siempre es la primera llave del diccionario)
        depot_id: int = list(nodes.keys())[0]

        # Aislamos los IDs de los clientes excluyendo el deposito
        clientes_ids: list = [n for n in nodes.keys() if n != depot_id]
        
        # Extraemos variables clave y ejecutamos el algoritmo K-Means
        etiquetas, clientes_ids = AngularClusteringManager._ejecutar_kmeans(nodes, demands, capacity, k_clusters, depot_id, clientes_ids)
        
        # Agrupamos los nodos en diccionarios basandonos en las etiquetas del algoritmo
        zonas_brutas: dict = AngularClusteringManager._agrupar_zonas(etiquetas, clientes_ids, nodes, demands, depot_id)
        
        # Instanciamos los objetos del problema CVRP listos para el PSO
        problemas_listos: list = AngularClusteringManager._instanciar_problemas(zonas_brutas, capacity, truck_penalty)
        
        # Retornamos la lista final
        return problemas_listos

    # Sobreescribimos la funcion interna del calculo K-Means para recibir demandas y capacidad
    @staticmethod
    def _ejecutar_kmeans(nodes: dict, demands: dict, capacity: int, k_clusters: int, depot_id: int, clientes_ids: list) -> tuple:
        
        # Extraemos las coordenadas del deposito para usarlas como centro del reloj
        depot_x, depot_y = nodes[depot_id]
        
        # Lista para guardar los datos transformados y normalizados
        datos_entrenamiento: list = AngularClusteringManager._preparar_datos_entrenamiento(
            nodes,
            demands,
            capacity,
            depot_x,
            depot_y,
            clientes_ids,
        )
            
        # Inicializamos y ejecutamos el algoritmo K-Means
        kmeans: KMeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
        etiquetas: np.ndarray = kmeans.fit_predict(datos_entrenamiento)
        
        # Devolvemos las etiquetas y la lista de IDs respetando el contrato de la clase padre
        return etiquetas, clientes_ids

    # Funcion interna para transformar coordenadas y demandas a escala [0, 1] en 5 dimensiones
    @staticmethod
    def _preparar_datos_entrenamiento(nodes: dict, demands: dict, capacity: int, depot_x: int, depot_y: int, clientes_ids: list) -> list:
        
        # Calculamos el promedio ideal usando logica de divisiones (Logaritmo)
        demanda_maxima: int = max([demands[c] for c in clientes_ids])
        n: int = math.ceil(math.log2(capacity / demanda_maxima))
        promedio_ideal: float = capacity / (2 ** n)
        
        # Pre-calculamos las demandas falsas para poder normalizarlas
        demandas_falsas = [abs(demands[c] - promedio_ideal) for c in clientes_ids]
        max_demanda_falsa: float = max(demandas_falsas)

        # Lista para guardar los datos transformados
        datos_entrenamiento: list = []
        
        # Iteramos por cada cliente para preparar su vector de 5 dimensiones
        for cliente in clientes_ids:
            
            # Posicion real del cliente
            x, y = nodes[cliente]
            
            # Calculamos la distancia relativa al deposito para el angulo
            dx: float = x - depot_x
            dy: float = y - depot_y
            angulo_radianes: float = math.atan2(dy, dx)
            
            # Guardamos el angulo desde el deposito (seno u coseno) en escala 0, 1
            cos_y_norm: float = (math.cos(angulo_radianes) + 1.0) / 2.0
            sin_x_norm: float = (math.sin(angulo_radianes) + 1.0) / 2.0
            
            # Calculamos la demanda falsa (complementaria)
            demanda_falsa: float = abs(demands[cliente] - promedio_ideal)
            demanda_falsa_norm: float = demanda_falsa / max_demanda_falsa
            
            # Añadimos el nuevo cliente
            datos_entrenamiento.append([sin_x_norm, cos_y_norm, demanda_falsa_norm])
        
        # Devolvemos los datos para el clustering
        return datos_entrenamiento