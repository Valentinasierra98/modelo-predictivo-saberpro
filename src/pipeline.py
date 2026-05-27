import os
from src.config import BaseConfig
from src.data import DataProcessor
from src.Models.train import TrainModels

def run_pipeline():
    modelo_path = os.path.join(BaseConfig.BASE_DIR, "models", "modelo_PUNT_GLOBAL.pkl")

    if os.path.exists(modelo_path):
        print("Pipeline ya existe, saltando entrenamiento")
        return

    print("Ejecutando pipeline completo...")

    BaseConfig()
    DataProcessor.ejecutar_acp_simple()
    TrainModels.entrenar_modelos()

    print("Pipeline listo")