# Librerias a usar
import math  # Modulo matematico para calcular distancias
import numpy as np  # Manejo eficiente de vectores numericos

from inspyred import benchmarks, ec, swarm  # Modulos de inspyred
from src.common.problem import CVRPProblem  # Clase que define el problema a resolver


# Definimos la clase base del problema CVRP adaptada para PSO
class BasePSO(benchmarks.Benchmark):

    # Constructor que inicializa el problema base
    def __init__(self, datos_problema: CVRPProblem=None, num_soluciones_corregir=None):
        
        # Almacenamos la instancia original del problema CVRP
        self.datos_problema: CVRPProblem = datos_problema

        # Vemos si nos dan un problema
        if datos_problema is not None:

            # Calculamos la dimension restando el deposito al conteo total
            self.dimension: int = len(datos_problema.node_ids) - 1
            
            # Inicializamos la clase padre con la dimension calculada
            benchmarks.Benchmark.__init__(self, self.dimension)

        # Si no nos lo dan, iniciamos a 0
        else:
            self.dimension: int = 0
        
        # Configuramos los limites continuos para el vector espacial (0.0 a 1.0)
        self.bounder: ec.Bounder = ec.Bounder(0.0, 1.0)
        
        # Desactivamos la maximizacion ya que buscamos la menor distancia
        self.maximize: bool = False
        
        # Asignamos el porcentaje o numero de soluciones que corregiremos con 2-opt
        self.num_soluciones_corregir = num_soluciones_corregir

    # Funcion que asigna un problema al algoritmo
    def asignar_problema(self, problema: CVRPProblem) -> None:

        # Asignamos el problema
        self.datos_problema: CVRPProblem = problema

        # Asignamos la dimension del problema
        self.dimension: int = len(problema.node_ids) - 1

        # Inicializamos la clase padre con la dimension calculada
        benchmarks.Benchmark.__init__(self, self.dimension)

        # Asignamos que minimizamos
        self.maximize: bool = False

    # Funcion que devuelve los parametros de la configuracion
    def get_params_configuracion(self) -> dict:

        # Lo devolvemos en un dict
        return {
            'num_soluciones_corregir': self.num_soluciones_corregir
        }

    # Funcion que devuelve el fenotipo de una particula
    def get_ruta_particula(self, particula: ec.Individual) -> list:

        # Obtenemos el fenotipo
        fenotipo_particula: list = self.decodificar_spv(particula.candidate)

        # Aplicamos el 2-opt si lo piden
        if self.num_soluciones_corregir:
            fenotipo_particula: list = self.aplicar_2_opt_ruta(fenotipo_particula)

        # Devolvemos el fenotipo (pasamos de genotipo a fenotipo)
        return fenotipo_particula

    # Funcion que configura el pso dependiendo de los datos
    def config_pso(self, algoritmo: swarm.PSO) -> None:
        
        # Le ponemos topologia anillo como base
        algoritmo.topology = swarm.topologies.ring_topology

    # Funcion generadora de nuevas particulas continuas
    def generator(self, random, args: dict) -> list:
        # Retornamos un vector dimensional con valores aleatorios entre 0 y 1
        return [random.uniform(0.0, 1.0) for _ in range(self.dimension)]

    # Funcion que evalua los individuos con su fitness
    def evaluator(self, candidates: list, args: dict) -> list:
        
        # Obtenemos los costos reales sin ninguna penalizacion
        fitness_crudo: list = []
        rutas_reales: list = []

        # Procesamos cada particula aplicando SPV (pasar de ghenotype a phenotype)
        for candidato in candidates:
            
            # Convertimos el vector real en una ruta discreta (SPV)
            ruta_bruta: list = self.decodificar_spv(candidato)
            rutas_reales.append(ruta_bruta)
            
            # Obtenemos la distancia de la ruta ya optimizada
            costo: float = self.datos_problema.evaluate_route_distance(ruta_bruta)
            
            # Guardamos el costo final
            fitness_crudo.append(costo)

        # Aplicamos 2-opt a los n mejores si corresponde
        self._aplicar_2_opt_global(fitness_crudo, rutas_reales)

        # Lo devolvemos
        return fitness_crudo

    # Funcion que devuelve los candidatos a devolver
    def _obtener_n_mejores(self, n_candidatos: int) -> int:

        # Si el usuario nos pasa 0, 0.0 o None, apagamos el 2-opt
        if not self.num_soluciones_corregir:
            return 0

        # Comprobamos si es un porcentaje (float)
        if isinstance(self.num_soluciones_corregir, float):

            # Calculamos la fraccion y aseguramos que al menos devuelva 3 (el minimo pedido)
            return max(3, int(n_candidatos * self.num_soluciones_corregir))
        
        # Comprobamos si es un numero exacto (int)
        elif isinstance(self.num_soluciones_corregir, int):

            # Devolvemos el numero de soluciones (minimo, el numero de candidatos. No recomendado)
            return min(n_candidatos, self.num_soluciones_corregir)
            
        # Por defecto sacamos 0
        return 0

    # Funcion que gestiona la aplicacion del 2-opt intra-ruta a los mejores
    def _aplicar_2_opt_global(self, fitness_crudo: list, rutas_reales: list) -> None:
        
        # Identificamos a cuantos debemos aplicar el 2-opt
        n_mejores: int = self._obtener_n_mejores(len(fitness_crudo))
        
        # Vemos si aplica usar opt
        if n_mejores > 0:

            # Obtenemos los indices de los N mejores
            mejores_candidatos: np.ndarray = np.argsort(fitness_crudo)[:n_mejores]
            
            # Recorremos los candidatos
            for candidato in mejores_candidatos:

                # Limpiamos la ruta
                ruta_limpia: list = self.aplicar_2_opt_ruta(rutas_reales[candidato])
                
                # Actualizamos la ruta real y recalculamos su coste
                rutas_reales[candidato] = ruta_limpia
                fitness_crudo[candidato] = self.datos_problema.evaluate_route_distance(ruta_limpia)

    # Funcion que divide el Tour Gigante en camiones individuales para limpiarlos
    def aplicar_2_opt_ruta(self, ruta: list) -> list:
        
        # Extraemos las posiciones de los depositos en la ruta
        indices_deposito: list = [i for i, x in enumerate(ruta) if x == self.datos_problema.depot_id]
        
        # Inicializamos la ruta optimizada
        ruta_optimizada: list = []
        
        # Iteramos por cada sub-ruta (cada viaje de un camion individual)
        for i in range(len(indices_deposito) - 1):
            
            # Extraemos el segmento (ruta) actual
            inicio: int = indices_deposito[i]
            fin: int = indices_deposito[i + 1]
            sub_ruta: list = ruta[inicio:fin + 1]
            
            # Calculamos la sub-ruta
            sub_ruta_limpia: list = self._optimizar_subruta_2_opt(sub_ruta)
                            
            # Añadimos la sub-ruta optimizada (sin duplicar depositos intermedios)
            if i == 0:
                ruta_optimizada.extend(sub_ruta_limpia)
            else:
                ruta_optimizada.extend(sub_ruta_limpia[1:])
        
        # Devolvemos la ruta optimizada
        return ruta_optimizada
    
    # Funcion aislada que realiza la busqueda local 2-opt en un solo vehiculo
    def _optimizar_subruta_2_opt(self, sub_ruta: list) -> list:
        
        # Banderas para controlar la mejora
        mejora: bool = True
        
        # Bucle hasta que no haya mas mejoras en este camión
        while mejora:
            mejora = False
            
            # Iteramos por pares de aristas para buscar cruces
            for j in range(1, len(sub_ruta) - 2):
                for k in range(j + 1, len(sub_ruta) - 1):
                    
                    # Si invertir el segmento acorta la distancia, lo aplicamos
                    if self._calcula_mejora_2_opt(sub_ruta, j, k):
                        sub_ruta[j:k + 1] = reversed(sub_ruta[j:k + 1])
                        mejora = True

        # Devolvemos la sub-ruta
        return sub_ruta
    
    # Funcion para evaluar si un cambio 2-opt reduce la distancia
    def _calcula_mejora_2_opt(
        self,
        sub_ruta: list,
        nodo_i: int,
        nodo_j: int
    ) -> bool:

        # Nodos implicados
        nodo_primero: int = sub_ruta[nodo_i - 1]
        nodo_segundo: int = sub_ruta[nodo_i]
        nodo_tercero: int = sub_ruta[nodo_j]
        nodo_cuarto: int = sub_ruta[nodo_j + 1]
        
        # Distancia original de las dos aristas (usando los indices corregidos)
        distancia_original: float = self._distancia_nodos(nodo_primero, nodo_segundo) + self._distancia_nodos(nodo_tercero, nodo_cuarto)
        
        # Distancia nueva si cruzaramos las aristas (usando los indices corregidos)
        distancia_nueva: float = self._distancia_nodos(nodo_primero, nodo_tercero) + self._distancia_nodos(nodo_segundo, nodo_cuarto)
        
        # Retorna True si la nueva distancia es mas corta
        return distancia_nueva < distancia_original
    
    # Funcion auxiliar para obtener la distancia entre dos nodos reales
    def _distancia_nodos(self, nodo_a: int, nodo_b: int) -> float:
        
        # Centralizamos aqui el ajuste de indices
        id_nodo_a: int = self.datos_problema.node_ids.index(nodo_a)
        id_nodo_b: int = self.datos_problema.node_ids.index(nodo_b)
        
        # Retornamos la distancia consultando la matriz precalculada
        return self.datos_problema.distance_matrix[id_nodo_a][id_nodo_b]

    # Funcion interna para transformar continuos a discretos
    def decodificar_spv(self, candidato: list) -> list:
        
        # Obtenemos los indices que ordenarian el arreglo ascendentemente
        indices_ordenados: np.ndarray = np.argsort(candidato)
        
        # Filtramos los nodos reales excluyendo el deposito central
        clientes: list = [nodo for nodo in self.datos_problema.node_ids if nodo != self.datos_problema.depot_id]
        
        # Mapeamos los clientes usando los indices para crear la secuencia
        tour_gigante: list = [clientes[i] for i in indices_ordenados]
        
        # Inicializamos el viaje partiendo de la ubicacion del deposito
        ruta_final: list = [self.datos_problema.depot_id]
        
        # Inicializamos la variable de capacidad consumida
        carga_actual: int = 0
        
        # Simulamos la entrega iterando por la secuencia construida
        for cliente in tour_gigante:
            
            # Consultamos el peso asignado al cliente actual
            demanda: int = self.datos_problema.demands[cliente]
            
            # Validamos si agregar este peso rompe las reglas del vehiculo
            if carga_actual + demanda > self.datos_problema.capacity:
                
                # Cerramos el viaje obligando al vehiculo a retornar
                ruta_final.append(self.datos_problema.depot_id)
                
                # Restablecemos la capacidad a cero para la nueva furgoneta
                carga_actual = 0
                
            # Agregamos la parada al recorrido vigente
            ruta_final.append(cliente)
            
            # Incrementamos la capacidad total con la carga reciente
            carga_actual += demanda
            
        # Comprobamos si el ultimo punto guardado fue distinto a la central
        if ruta_final[-1] != self.datos_problema.depot_id:
            
            # Aseguramos la legalidad forzando el retorno final
            ruta_final.append(self.datos_problema.depot_id)
            
        # Retornamos la estructura valida lista para su evaluacion de coste
        return ruta_final