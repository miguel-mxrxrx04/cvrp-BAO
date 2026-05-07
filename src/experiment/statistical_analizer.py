# Librerias a usar
import numpy as np  # Libreria para calculos numericos avanzados y medianas
import pandas as pd  # Libreria para manipulacion de estructuras de datos tabulares
import seaborn as sns  # Libreria basada en matplotlib para graficos estadisticos mas esteticos
import networkx as nx  # Libreria especializada para crear y visualizar grafos de nodos
import matplotlib.pyplot as plt  # Libreria estandar para generar graficos 2D

from scipy.stats import wilcoxon  # Funcion para evaluar diferencias entre 2 algoritmos (1 vs 1)
from stac.nonparametric_tests import shaffer_multitest  # Test Post-Hoc para ver quien gana a quien (N vs N)
from stac.nonparametric_tests import friedman_aligned_ranks_test  # Test no parametrico para multiples algoritmos (N vs N)


# Clase encargada de orquestar las pruebas estadisticas y sus visualizaciones visuales
class StatisticalAnalyzer:

    # Constructor de la clase que recibe el dataframe global unificado
    def __init__(self, experimentos_df: pd.DataFrame, alpha: float=0.05):

        # Asignamos el dataframe crudo al atributo de la instancia
        self.datos_crudos: pd.DataFrame = experimentos_df

        # Fijamos el nivel de significancia estandar del 5% para los P-Values
        self.alpha: float = alpha

        # Generamos los datos colapsados (las medianas por mapa) necesarios para Multi-Problem
        self.datos_multiproblema: pd.DataFrame = self._preparar_datos_multiproblema()

    # Metodo privado que aplica la regla del Multi-Problem (calcular la mediana por dataset)
    def _preparar_datos_multiproblema(self) -> pd.DataFrame:
        
        # Agrupamos los datos crudos por Algoritmo y por el Mapa (experiment_id)
        agrupacion = self.datos_crudos.groupby(['algorithm', 'experiment_id'])
        
        # Calculamos la mediana de todas las metricas para cada grupo
        datos_mediana: pd.DataFrame = agrupacion.median(numeric_only=True).reset_index()
        
        # Devolvemos el nuevo dataframe condensado
        return datos_mediana

    # Funcion publica para comparar graficamente el rendimiento global usando Diagramas de Caja
    def dibujar_boxplot_global(self, metrica: str = 'mean_fitness', titulo: str = 'Comparativa Global de Fitness') -> None:
        
        # Creamos el lienzo del grafico especificando sus dimensiones
        plt.figure(figsize=(10, 6))
        
        # Configuramos el estilo de fondo mediante seaborn
        sns.set_theme(style="whitegrid")
        
        # Dibujamos el boxplot (Corregido para evitar el FutureWarning de Seaborn)
        sns.boxplot(x="algorithm", y=metrica, data=self.datos_multiproblema, hue="algorithm", palette="Set2", legend=False)
        
        # Añadimos el titulo principal al grafico
        plt.title(titulo, fontsize=14, fontweight='bold')
        
        # Ajustamos las etiquetas de los ejes
        plt.xlabel("Algoritmo Evaluado", fontsize=12)
        plt.ylabel(metrica, fontsize=12)
        
        # Mostramos el grafico en pantalla
        plt.show()

    # Funcion que ejecuta el test estadistico para 2 algoritmos (1 vs 1)
    def comparar_1_vs_1(self, alg_a: str, alg_b: str, metrica: str = 'mean_fitness', es_minimizar: bool = True) -> None:
        
        # Filtramos los datos pertenecientes al primer algoritmo
        datos_a: pd.Series = self.datos_multiproblema[self.datos_multiproblema['algorithm'] == alg_a][metrica]
        
        # Filtramos los datos pertenecientes al segundo algoritmo
        datos_b: pd.Series = self.datos_multiproblema[self.datos_multiproblema['algorithm'] == alg_b][metrica]
        
        # Ejecutamos el test de Wilcoxon para muestras pareadas
        res = wilcoxon(datos_a, datos_b)
        
        # Imprimimos la cabecera de los resultados
        print(f'--- Test 1 vs 1 (Wilcoxon): {alg_a} vs {alg_b} [{metrica}] ---')
        
        # Verificamos si el p-value es menor a nuestro alpha
        if res.pvalue < self.alpha:
            
            # Hay diferencia estadistica confirmada
            print(f'Resultado: La diferencia ES estadísticamente significativa (p-value: {res.pvalue:.5f}).')
            
            # Determinamos quien es mejor basandonos en la mediana y si buscamos minimizar o maximizar
            tiene_menor_valor = datos_a.median() < datos_b.median()
            if (tiene_menor_valor and es_minimizar) or (not tiene_menor_valor and not es_minimizar):
                print(f'Veredicto: {alg_a} es superior a {alg_b}.')
            else:
                print(f'Veredicto: {alg_b} es superior a {alg_a}.')
        else:
            
            # No hay evidencia suficiente para diferenciarlos
            print(f'Resultado: No hay evidencia estadística para afirmar que son distintos (p-value: {res.pvalue:.5f}).')

    # Funcion experta que ejecuta el test de Friedman y Shaffer para comparar 3 o mas algoritmos
    def comparar_n_vs_n(self, metrica: str = 'mean_fitness', es_minimizar: bool=True) -> None:

        # Alineamos los datos por mapa (experiment_id) como filas y algoritmos como columnas
        df_pivot = self.datos_multiproblema.pivot(index='experiment_id', columns='algorithm', values=metrica)
        
        # Eliminamos cualquier mapa (fila) que no tenga datos en todos los algoritmos
        df_pivot = df_pivot.dropna()
        
        # Obtenemos la lista unica de algoritmos involucrados
        nombres_algs: list = list(df_pivot.columns)
        
        # Creamos un diccionario para mapear los nombres a posiciones numericas
        names_pos: dict = dict(zip(nombres_algs, range(len(nombres_algs))))
        
        # Creamos una lista para almacenar las series de datos de cada algoritmo en orden
        series_datos: list = [df_pivot[alg].values for alg in nombres_algs]
            
        # Ejecutamos el test de Rangos Alineados de Friedman pasando los arreglos desempaquetados
        _, p_value, rankings, pivots = friedman_aligned_ranks_test(*series_datos)
        
        # Imprimimos la cabecera
        print(f'--- Test N vs N (Friedman Aligned Ranks): Metrica [{metrica}] ---')
        
        # Evaluamos el p-value global del Friedman
        if p_value < self.alpha:
            
            # Hay diferencias globales, procedemos con el test Post-Hoc de Shaffer
            print('El Test de Friedman indica que AL MENOS UN ALGORITMO es diferente.')
            
            # Emparejamos los nombres con sus pivotes calculados por Friedman
            d = dict(zip(nombres_algs, pivots))
            
            # Ejecutamos la correccion de Shaffer
            comp, _, _, adpval = shaffer_multitest(d)
            
            # Iteramos sobre los emparejamientos resultantes
            for i, apv in enumerate(adpval):
                
                # Dividimos el string de comparacion para obtener los nombres
                chunks = comp[i].split('vs')
                name_l = chunks[0].strip()
                name_r = chunks[1].strip()
                
                # Verificamos si este emparejamiento especifico tiene diferencia estadistica
                if apv < self.alpha:
                    
                    # Determinamos quien tiene mejor rango (menor rango es mejor)
                    tiene_menor_rango = rankings[names_pos[name_l]] < rankings[names_pos[name_r]]
                    
                    # Interpretamos quien gana segun si buscamos minimizar o maximizar
                    if (tiene_menor_rango and es_minimizar) or (not tiene_menor_rango and not es_minimizar):
                        print(f'-> {name_l} es estadísticamente MEJOR que {name_r} (p-adj: {apv})')
                    else:
                        print(f'-> {name_r} es estadísticamente MEJOR que {name_l} (p-adj: {apv})')
                else:
                    
                    # Son equivalentes estadisticamente
                    print(f'-> No hay diferencia significativa entre {name_l} y {name_r} (Empate)')
        else:
            
            # Ningun algoritmo destaca sobre los demas
            print(f'El Test de Friedman indica que NO hay diferencia significativa entre ninguno de los algoritmos (p-value: {p_value:.5f}).')