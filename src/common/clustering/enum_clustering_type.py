# Libreria a usar
from enum import Enum  # Para crear la clase de opciones cerradas
from src.common.clustering.clustering_manager import ClusteringManager  # Creador de clusterings basados en demandas complementarias y posicion en el mapa
from src.common.clustering.angular_clustering_manager import AngularClusteringManager  # Creador de clusterings basados en demandas complementarias y angulos
from src.common.clustering.complete_clustering_manager import CompleteClusteringManager  # Creador de clusterings combinando ambas ideas


# Enum para saber los tipos de clustering
class ClusteringTypes(Enum):
    
    # Clustering basado en posicion
    NORMAL = ClusteringManager
    
    # Clustering basado en angulos
    ANGULAR = AngularClusteringManager

    # Clustering que combina ambos
    COMPLETE = CompleteClusteringManager