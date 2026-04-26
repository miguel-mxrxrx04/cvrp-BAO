# Librerias a usar
import math  # Modulo matematico para calculos de redondeo y absolutos
import numpy as np  # Manejo eficiente de vectores numericos
from sklearn.cluster import KMeans  # Modulo para aplicar el algoritmo K-Means

from src.common.clustering.clustering_problem import ClusterCVRPProblem  # Clase hija que vamos a instanciar


# Clase estatica encargada de dividir el mapa en sub-problemas
class ClusteringManager:
    
    # Funcion para calcular la cantidad de clústers ideal basada en la carga del mapa
    @staticmethod
    def calcular_k_ideal(num_clientes: int, max_clientes_por_cluster: int=40) -> int:
        
        # Calculamos el K dividiendo y redondeando hacia arriba
        k_calculado: int = math.ceil(num_clientes / max_clientes_por_cluster)
        
        # Aseguramos que como minimo siempre devuelva 1 cluster
        return max(1, k_calculado)

    # Funcion principal que orquesta la division llamando a los metodos privados
    @staticmethod
    def generar_sub_problemas(nodes: dict, demands: dict, capacity: int, k_clusters: int, truck_penalty: float=2) -> list:
        
        # Extraemos el ID del deposito (siempre es la primera llave del diccionario)
        depot_id: int = list(nodes.keys())[0]

        # Aislamos los IDs de los clientes excluyendo el deposito
        clientes_ids: list = [n for n in nodes.keys() if n != depot_id]
        
        # Extraemos variables clave y ejecutamos el algoritmo K-Means
        etiquetas, clientes_ids = ClusteringManager._ejecutar_kmeans(nodes, k_clusters, clientes_ids)
        
        # Agrupamos los nodos en diccionarios basandonos en las etiquetas del algoritmo
        zonas_brutas = ClusteringManager._agrupar_zonas(etiquetas, clientes_ids, nodes, demands, depot_id)
        
        # Instanciamos los objetos del problema CVRP listos para el PSO
        problemas_listos = ClusteringManager._instanciar_problemas(zonas_brutas, capacity, truck_penalty)
        
        # Retornamos la lista final
        return problemas_listos

    # Funcion interna para realizar la prediccion geometrica/demanda
    @staticmethod
    def _ejecutar_kmeans(nodes: dict, k_clusters: int, clientes_ids: list) -> tuple:
        
        # Lista para guardar los datos transformados
        datos_entrenamiento: list = []
        
        # Iteramos por cada cliente para preparar sus coordenadas
        for cliente in clientes_ids:
            
            # Posicion del cliente
            x, y = nodes[cliente]
            
            # Añadimos el nuevo cliente (con la demanda cambiada para agrupar clientes con demandas desiguales)
            datos_entrenamiento.append([x, y])
            
        # Inicializamos y ejecutamos el algoritmo K-Means
        kmeans: KMeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
        etiquetas: np.ndarray = kmeans.fit_predict(datos_entrenamiento)
        
        # Devolvemos las etiquetas, clientes y la id del deposito
        return etiquetas, clientes_ids

    # Funcion interna para separar los diccionarios originales en sub-zonas
    @staticmethod
    def _agrupar_zonas(etiquetas: np.ndarray, clientes_ids: list, nodes: dict, demands: dict, depot_id: int) -> dict:
        
        # Diccionario temporal para agrupar los nodos
        zonas_brutas: dict = {}
        
        # Iteramos por las etiquetas generadas para reconstruir los diccionarios
        for idx, cluster_id in enumerate(etiquetas):
            
            # Recuperamos el ID real del cliente procesado
            cliente_id = clientes_ids[idx]
            
            # Si el cluster es nuevo, lo inicializamos asegurando que incluya el deposito
            if cluster_id not in zonas_brutas:
                zonas_brutas[cluster_id] = {
                    'nodes': {depot_id: nodes[depot_id]},
                    'demands': {depot_id: demands.get(depot_id, 0)},
                    'carga_total': 0
                }
                
            # Añadimos los datos del cliente a su cluster correspondiente
            zonas_brutas[cluster_id]['nodes'][cliente_id] = nodes[cliente_id]
            zonas_brutas[cluster_id]['demands'][cliente_id] = demands[cliente_id]
            
            # Sumamos la demanda real a la carga total de esta zona
            zonas_brutas[cluster_id]['carga_total'] += demands[cliente_id]
        
        # Devolvemos las zonas (clusters)
        return zonas_brutas

    # Funcion interna para inicializar las clases de los problemas
    @staticmethod
    def _instanciar_problemas(zonas_brutas: dict, capacity: int, truck_penalty: int) -> list:
        
        # Lista final donde guardaremos los objetos
        problemas_listos: list = []
        
        # Iteramos por cada zona agrupada para crear las instancias
        for data in zonas_brutas.values():
            
            # Calculamos el minimo estricto de camiones teoricos para esta zona
            m_minimo: int = math.ceil(data['carga_total'] / capacity)
            
            # Instanciamos el problema especializado pasandole los datos y el limite
            sub_problema = ClusterCVRPProblem(
                nodes=data['nodes'],
                demands=data['demands'],
                capacity=capacity,
                m_minimo=m_minimo,
                truck_penalty=truck_penalty
            )
            
            # Lo añadimos a la lista
            problemas_listos.append(sub_problema)
        
        # Devolvemos los problemas
        return problemas_listos