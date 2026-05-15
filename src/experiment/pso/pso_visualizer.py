# Librerias a usar
import numpy as np  # Manejo eficiente de vectores numéricos
import matplotlib.pyplot as plt  # Para la generacion de graficos

from src.common.problem import CVRPProblem  # Clase que usaremos para las estadisticas (tendra los resultados de las ejecuciones)


# Clase encargada de toda la visualizacion de datos
class VisualizadorCVRP:
    
    # Constructor que inicializa el visualizador con el mapa del problema
    def __init__(self, problema: CVRPProblem):
        
        # Almacenamos la instancia original para acceder a las coordenadas
        self.problema = problema
        
        # Definimos una paleta de 20 colores para diferenciar los vehiculos
        self.paleta_colores = plt.cm.tab20.colors

    # Funcion para replicar el grafico doble de Evolucion y Diversidad
    def dibujar_evolucion(self, historico_fitness_mejor: list, historico_fitness_media: list, historico_diversidad: list) -> None:
        
        # Creamos un lienzo con dos subgraficos lado a lado (1 fila, 2 columnas)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Grafico Izquierdo: Curva de Convergencia (Fitness)
        ax1.plot(historico_fitness_mejor, color='#2ca02c', linewidth=2, label='Best Fitness')
        ax1.plot(historico_fitness_media, color='#1f77b4', linewidth=2, linestyle='--', alpha=0.8, label='Mean fitnees')
        ax1.set_title('Fitness over Generations', fontsize=12)
        ax1.set_xlabel('Generation', fontsize=10)
        ax1.set_ylabel('Fitness (Distance)', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        # Grafico Derecho: Caida de Diversidad
        ax2.plot(historico_diversidad, color='#ff7f0e', linewidth=2)
        ax2.set_title('Diversity over Generations', fontsize=12)
        ax2.set_xlabel('Generation', fontsize=10)
        ax2.set_ylabel('Diversity (Std Dev)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Titulo general para la ventana completa
        fig.suptitle('Fitness and Diversity of Evolutionary Algorithm', fontsize=14, fontweight='bold')
        
        # Ajustamos los margenes para que no se superpongan los textos
        plt.tight_layout()
        plt.show()

    # Funcion para analizar como de agrupadas estan las particulas al final
    def dibujar_distribucion_poblacion(self, fitness_poblacion: list) -> None:
        
        # Creamos un lienzo con dos subgraficos lado a lado
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Grafico Izquierdo: Histograma de Frecuencias 
        ax1.hist(
            fitness_poblacion,
            bins=15,
            color='#2ca02c',
            edgecolor='black',
            alpha=0.7
        )
        ax1.set_title('Distribución del Fitness (Histograma)', fontsize=12)
        ax1.set_xlabel('Costo de la Ruta', fontsize=10)
        ax1.set_ylabel('Número de Partículas', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Grafico Derecho: Boxplot (Diagrama de Caja) para outliers
        ax2.boxplot(
            fitness_poblacion,
            vert=False,
            patch_artist=True, 
            boxprops={
                'facecolor': '#d62728',
                'color': 'black'
            },
            medianprops={
                'color': 'white',
                'linewidth': 2
            }
        )
        ax2.set_title('Dispersión del Fitness (Boxplot)', fontsize=12)
        ax2.set_xlabel('Costo de la Ruta', fontsize=10)
        ax2.set_yticks([])
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # Titulo general
        fig.suptitle('Análisis Estadístico de la Población Final', fontsize=14, fontweight='bold')
        
        # Ajustamos y mostramos
        plt.tight_layout()
        plt.show()

    # Funcion para dibujar el mapa fisico con las rutas sin cruces
    def dibujar_rutas(self, ruta_decodificada: list, titulo: str='Mapa de Rutas Optimizadas') -> None:
        
        # Creamos un lienzo grande para el mapa
        plt.figure(figsize=(12, 8))
        
        # Dibujamos todos los clientes como puntos de fondo grises
        for nodo_id, coords in self.problema.nodes.items():
            
            # Omitimos el deposito para pintarlo diferente despues
            if nodo_id != self.problema.depot_id:
                plt.scatter(coords[0], coords[1], c='gray', s=30, alpha=0.5)
                
        # Obtenemos y dibujamos el deposito central como un cuadrado rojo grande
        coord_deposito: tuple = self.problema.nodes[self.problema.depot_id]
        plt.scatter(coord_deposito[0], coord_deposito[1], c='red', marker='s', s=150, label='Depósito Central', zorder=5)
        
        # Extraemos las posiciones del deposito para separar los viajes
        indices_deposito: list = [i for i, x in enumerate(ruta_decodificada) if x == self.problema.depot_id]
        
        # Contador para asignar colores secuenciales
        num_camion: int = 0
        
        # Iteramos extrayendo el viaje individual de cada vehiculo
        for i in range(len(indices_deposito) - 1):
            
            # Cortamos la lista usando los indices
            inicio: int = indices_deposito[i]
            fin: int = indices_deposito[i+1]
            viaje: list = ruta_decodificada[inicio:fin+1]
            
            # Filtramos vehiculos vacios (rutas que son solo [1, 1])
            if len(viaje) > 2:
                
                # Extraemos listas de coordenadas X e Y
                coordenadas_x: list = [self.problema.nodes[nodo][0] for nodo in viaje]
                coordenadas_y: list = [self.problema.nodes[nodo][1] for nodo in viaje]
                
                # Asignamos el color ciclico segun el indice
                color_camion: tuple = self.paleta_colores[num_camion % len(self.paleta_colores)]
                
                # Trazamos la ruta solida
                plt.plot(coordenadas_x, coordenadas_y, c=color_camion, linewidth=2, alpha=0.8, marker='o', markersize=5, label=f'Vehículo {num_camion + 1}')
                
                # Incrementamos el contador
                num_camion += 1
                
        # Configuracion estetica final del mapa
        plt.title(titulo, fontsize=15, fontweight='bold')
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Desplazamos la leyenda fuera del grafico para no tapar los caminos
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Ajustamos y mostramos
        plt.tight_layout()
        plt.show()