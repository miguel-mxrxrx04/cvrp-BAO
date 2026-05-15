# Librerias a usar
import math  # Modulo matematico para calcular distancias entre coordenadas
from typing import Optional  # Para el tipado de variables que pueden ser nulas o vacias


# Clase encargada de juntar rutas terminadas para reducir el numero total de camiones
class JoinRoutes:

    # Funcion principal que hace la reduccion de rutas hasta cumplir el limite
    @staticmethod
    def consolidar_rutas(
        rutas_pso: list,
        demands: dict,
        capacity: int,
        limite_camiones: int,
        depot_id: int,
        nodes: dict
    ) -> list:
        
        # Trabajamos sobre una copia de las rutas para no alterar el array original
        rutas_actuales: list = rutas_pso.copy()
        
        # Bucle que se ejecuta MIENTRAS estemos pasados del limite de camiones
        while len(rutas_actuales) > limite_camiones:
            
            # Calculamos la carga actual de todas las rutas en la lista
            datos_rutas: list = JoinRoutes._calcular_cargas(rutas_actuales, demands, depot_id)
            
            # Buscamos la pareja ideal que desperdicie la menor cantidad de espacio posible
            pareja_ideal: Optional[dict] = JoinRoutes._encontrar_mejor_empalme(datos_rutas, capacity)
            
            # Si encontramos una pareja compatible, ejecutamos el empalme
            if pareja_ideal is not None:
                idx_1: int = pareja_ideal['i']
                idx_2: int = pareja_ideal['j']
                
                # Generamos la nueva ruta fusionada
                nueva_ruta: list = JoinRoutes._empalmar_rutas(datos_rutas[idx_1]['ruta'], datos_rutas[idx_2]['ruta'], nodes)
                
                # Eliminamos las rutas viejas de la lista (borramos el indice mayor primero para no alterar el orden de la lista)
                rutas_actuales.pop(max(idx_1, idx_2))
                rutas_actuales.pop(min(idx_1, idx_2))
                
                # Añadimos la nueva super-ruta a la lista activa
                rutas_actuales.append(nueva_ruta)
                print(f'Rutas fusionadas. Camiones actuales: {len(rutas_actuales)}')
            
            # Si no hay pareja_ideal, es fisicamente imposible juntar mas rutas sin pasarse de la capacidad
            else:
                print(f'Limite fisico de capacidad alcanzado. Imposible reducir a {limite_camiones}.')
                break
                
        # Retornamos la lista de rutas final (optimizada en cantidad de camiones y sin cruces)
        return rutas_actuales

    # Funcion interna para calcular el peso total que lleva cada ruta devuelta por el PSO
    @staticmethod
    def _calcular_cargas(
        rutas: list,
        demands: dict,
        depot_id: int
    ) -> list:
        
        # Lista donde guardaremos los datos auditados de cada ruta
        datos_rutas: list = []
        
        # Iteramos por cada ruta devuelta por el PSO para auditar su carga real
        for i, ruta in enumerate(rutas):
            
            # Sumamos la demanda de los clientes (ignoramos el deposito porque su demanda es 0 o ficticia)
            carga: int = sum(demands[cliente] for cliente in ruta if cliente != depot_id)
            
            # Guardamos un diccionario con el indice original, los nodos de la ruta y su peso total
            datos_rutas.append({'index': i, 'ruta': ruta, 'carga': carga})
            
        # Retornamos la lista con todas las rutas y sus cargas calculadas
        return datos_rutas

    # Funcion interna para buscar que dos rutas encajan mejor en un solo camion (Tetris logistico)
    @staticmethod
    def _encontrar_mejor_empalme(datos_rutas: list, capacity: int) -> Optional[dict]:
        
        # Variable para guardar los indices de la mejor pareja (None si no hay parejas posibles)
        pareja_ideal: Optional[dict] = None
        
        # Inicializamos el espacio desperdiciado en infinito para buscar siempre el minimo
        menor_espacio_desperdiciado: float = float('inf')
        
        # Comparamos todas las rutas entre si para hallar la combinacion optima (O(n^2) pero rapido por ser pocas)
        for i in range(len(datos_rutas)):
            for j in range(i + 1, len(datos_rutas)):
                
                # Calculamos el peso teorico si juntaramos ambas rutas en un solo camion
                carga_combinada: int = datos_rutas[i]['carga'] + datos_rutas[j]['carga']
                
                # Verificamos si es un movimiento legal (la suma no supera la capacidad maxima del camion)
                if carga_combinada <= capacity:
                    
                    # Calculamos cuanto hueco quedaria vacio en ese camion (espacio desperdiciado)
                    espacio_sobrante: int = capacity - carga_combinada
                    
                    # Nos quedamos estrictamente con la pareja que llene mejor el camion (menor espacio sobrante)
                    if espacio_sobrante < menor_espacio_desperdiciado:
                        
                        # Actualizamos el nuevo record de espacio minimo desperdiciado
                        menor_espacio_desperdiciado = espacio_sobrante
                        
                        # Guardamos los indices de esta nueva mejor pareja
                        pareja_ideal = {'i': i, 'j': j}
                        
        # Retornamos el diccionario con los indices de las dos rutas elegidas (o None si no hubo exito)
        return pareja_ideal

    # Funcion interna para unir dos listas de rutas quitando cruces innecesarios y aplicando 2-opt
    @staticmethod
    def _empalmar_rutas(
        ruta_1: list,
        ruta_2: list,
        nodes: dict
    ) -> list:
        
        # Excluimos el ultimo elemento de la ruta 1 y el primer elemento de la ruta 2 para fusionarlas
        nueva_ruta_sucia: list = ruta_1[:-1] + ruta_2[1:]

        # Aplicamos el 2-opt que ahora vive dentro de esta misma clase
        nueva_ruta_limpia: list = JoinRoutes.aplicar_2_opt(nueva_ruta_sucia, nodes)
        
        # Devolvemos la super-ruta final optimizada geometricamente
        return nueva_ruta_limpia

    # Funcion principal para aplicar el algoritmo 2-opt y desenredar rutas
    @staticmethod
    def aplicar_2_opt(ruta: list, nodes: dict) -> list:
        
        # Variable para controlar si hubo mejoras en la iteracion actual
        mejora: bool = True
        
        # Copiamos la ruta para trabajar sobre ella sin modificar la original
        mejor_ruta: list = ruta.copy()
        
        # Bucle principal: repetimos mientras sigamos encontrando conexiones mas cortas
        while mejora:
            mejora = False
            
            # Iteramos por todos los pares de aristas posibles
            for i in range(1, len(mejor_ruta) - 2):
                for j in range(i + 1, len(mejor_ruta) - 1):
                    
                    # Calculamos si invertir este segmento nos ahorra kilometros
                    cambio_distancia: float = JoinRoutes._calcular_cambio_distancia(mejor_ruta, i, j, nodes)
                    
                    # Si el cambio es negativo (es decir, la nueva distancia es menor), hacemos el cambio
                    if cambio_distancia < -0.0001:
                        
                        # Invertimos el segmento de la ruta (Slicing con reverse)
                        mejor_ruta[i:j+1] = reversed(mejor_ruta[i:j+1])
                        
                        # Marcamos que hubo una mejora para volver a revisar toda la ruta desde cero
                        mejora = True
                        
        # Devolvemos la ruta final sin cruces
        return mejor_ruta

    # Funcion interna para calcular matematicamente si un cambio de aristas reduce la distancia
    @staticmethod
    def _calcular_cambio_distancia(
        ruta: list,
        i: int,
        j: int,
        nodes: dict
    ) -> float:
        
        # Identificamos los 4 nodos implicados en el posible cruce
        nodo_a_previo: int = ruta[i - 1]
        nodo_a: int = ruta[i]
        nodo_b: int = ruta[j]
        nodo_b_post: int = ruta[j + 1]
        
        # Calculamos la distancia de las conexiones actuales
        distancia_original: float = math.dist(nodes[nodo_a_previo], nodes[nodo_a]) + math.dist(nodes[nodo_b], nodes[nodo_b_post])
        
        # Calculamos la distancia si cruzáramos las aristas (invirtiendo el camino intermedio)
        distancia_nueva: float = math.dist(nodes[nodo_a_previo], nodes[nodo_b]) + math.dist(nodes[nodo_a], nodes[nodo_b_post])
        
        # Retornamos la diferencia (si es negativa, significa ahorro logistico)
        return distancia_nueva - distancia_original