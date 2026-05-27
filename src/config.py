"""
MÓDULO DE CONFIGURACIÓN GENERAL PROYECTO MLOPS - SABER PRO UAO
Este archivo centraliza las rutas globales del sistema y las variables
objetivo para evitar errores de rutas relativas entre computadoras.
"""

import os
from pathlib import Path

class BaseConfig:

    # ==============================================================================
    #  CONFIGURACIÓN DE RUTA RAÍZ Y CARPETAS (ESTÁNDAR MLOPS / DATAOPS)
    # ==============================================================================

    # BASE_DIR calcula automáticamente la carpeta principal del proyecto
    # Sirve para detectará la raíz.
    BASE_DIR = Path(__file__).resolve().parent.parent

    # RAW_DATA_PATH: Ubicación exacta del Excel original con las 11,229 filas (Datos crudos)
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "BASE_TRABAJO_GRADO.xlsx")

    # PROCESSED_DATA_PATH: Destino donde se guardará el archivo CSV limpio y listo para IA
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "data_limpia.csv")


    # ==============================================================================
    # VARIABLES OBJETIVO DE LA TESIS (TARGETS MULTI-SALIDA)
    # ==============================================================================

    # TARGET_COLUMNS: Lista con las 6 competencias del Saber Pro que el modelo va a predecir.
    # Al dejarlas blindadas aquí, el script de limpieza sabrá que NO debe borrarlas.
    TARGET_COLUMNS = [
        "PUNT_GLOBAL",                 # Puntaje general acumulado
        "MOD_RAZONA_CUANTITAT_PUNT",   # Módulo de Razonamiento Cuantitativo
        "MOD_LECTURA_CRITICA_PUNT",    # Módulo de Lectura Crítica
        "MOD_COMPETEN_CIUDADA_PUNT",   # Módulo de Competencias Ciudadanas
        "MOD_INGLES_PUNT",             # Módulo de Inglés
        "MOD_COMUNI_ESCRITA_PUNT"      # Módulo de Comunicación Escrita
    ]

    # ==============================================================================
    # HIPERPARÁMETROS DE ENTRENAMIENTO
    # ==============================================================================

    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 8
    RANDOM_STATE = 42
    PCA_N_COMPONENTS = 10

    print("Rutas y variables objetivo configuradas con éxito.")
