import time
import concurrent.futures
from src.pso.pso_algorithm import PSOAlgorithm
from src.pso.multimodal_techniques.sequential_pso import SequentialPSO

class ParallelClustering:

    @staticmethod
    def _resolver_zona_completa(
            zona_id: int, problema_local,
            parametros_pso: dict, parametros_config_pso: dict,
            depot_id: int, num_ejecuciones: int
        ) -> tuple:

        # 1. Creamos el motor UNA SOLA VEZ para que conserve su "suerte"
        motor_local: PSOAlgorithm = PSOAlgorithm(
            tamano_poblacion=parametros_pso['tamano_poblacion'],
            max_evaluaciones=parametros_pso['max_evaluaciones'],
            inercia=parametros_pso['inercia'],
            cognitivo=parametros_pso['cognitivo'],
            social=parametros_pso['social']
        )
        
        historial_rutas_zona = []
        camiones_por_iteracion = []
        
        # 2. Las 3 iteraciones se hacen AQUÍ ADENTRO
        for _ in range(num_ejecuciones):
            
            tecnica_local: SequentialPSO = SequentialPSO(
                datos_problema=problema_local,
                umbral_similitud=parametros_config_pso['umbral_similitud'],
                penalizacion=parametros_config_pso['penalizacion'],
                num_soluciones_corregir=parametros_config_pso['num_soluciones_corregir']
            )
            
            # Memoria local de esta zona
            tecnica_local.optimos_encontrados = historial_rutas_zona.copy()
            
            motor_local.clear() # Limpiamos enjambre, pero el motor sigue vivo
            motor_local.ejecutar(tecnica_local)
            
            mejor_particula = motor_local.mejores_soluciones[0]
            ruta_sucia: list = tecnica_local._decodificar_spv(mejor_particula.candidate)
            ruta_limpia: list = tecnica_local._aplicar_2_opt_ruta(ruta_sucia)
            
            # Guardamos exactamente tu ruta limpia como te gustaba
            historial_rutas_zona.append(ruta_limpia)
            
            # Troceamos los camiones
            camiones_troceados: list = []
            viaje_actual: list = [depot_id]
            for nodo in ruta_limpia[1:]:
                viaje_actual.append(nodo)
                if nodo == depot_id:
                    camiones_troceados.append(viaje_actual)
                    viaje_actual = [depot_id]
                    
            camiones_por_iteracion.append(camiones_troceados)
            
        return zona_id, camiones_por_iteracion

    @staticmethod
    def resolver_todo_en_paralelo(
            sub_problemas: list, parametros_pso: dict, 
            parametros_config_pso: dict, depot_id: int, num_ejecuciones: int
        ) -> list:
        
        # camiones_globales[0] tendrá los camiones de todas las zonas para la iteración 1
        camiones_globales = [ [] for _ in range(num_ejecuciones) ]
        
        with concurrent.futures.ProcessPoolExecutor() as ejecutor:
            futuros: list = [
                ejecutor.submit(
                    ParallelClustering._resolver_zona_completa,
                    i, problema, parametros_pso, parametros_config_pso, 
                    depot_id, num_ejecuciones
                )
                for i, problema in enumerate(sub_problemas)
            ]
            
            for futuro in concurrent.futures.as_completed(futuros):
                _, camiones_iteraciones = futuro.result()
                
                # Repartimos los camiones devueltos en su iteración global correspondiente
                for it in range(num_ejecuciones):
                    camiones_globales[it].extend(camiones_iteraciones[it])
                
        return camiones_globales