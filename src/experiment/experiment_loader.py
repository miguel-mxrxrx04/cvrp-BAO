# Librerias a usar
import pandas as pd  # Libreria para manipulacion y concatenacion de estructuras de datos tabulares
from os import listdir  # Funcion para listar el contenido de un directorio del sistema
from os.path import isfile, join  # Funciones para validacion y union segura de rutas de archivos


# Clase utilitaria para cargar y unificar los resultados de multiples experimentos
class ExperimentLoader:

    # Constructor de la clase que recibe la ruta del directorio de experimentos
    def __init__(self, experiment_folder: str):

        # Asignamos la ruta de la carpeta al atributo de la instancia
        self.experiment_folder: str = experiment_folder

        # Ejecutamos el metodo privado de carga y almacenamos el dataframe resultante
        self.experiment_data: pd.DataFrame = self._load_data()

    # Metodo privado encargado de leer y concatenar todos los archivos csv
    def _load_data(self) -> pd.DataFrame:

        # Generamos una lista de comprension con los nombres de archivo validos dentro del directorio
        files: list = [f for f in listdir(self.experiment_folder) if isfile(join(self.experiment_folder, f))]

        # Inicializamos una lista vacia para acumular los dataframes individuales
        data: list = []

        # Iteramos sobre cada archivo detectado en la carpeta
        for f in files:

            # Leemos el archivo csv actual y lo convertimos en un dataframe de pandas
            actual_data: pd.DataFrame = pd.read_csv(f'{self.experiment_folder}/{f}')

            # Agregamos el dataframe actual a la lista de acumulacion
            data.append(actual_data)
            
        # Concatenamos todos los dataframes recopilados y devolvemos la estructura unificada
        return pd.concat(data)