"""
MÓDULO DE ENTRENAMIENTO, EVALUACIÓN Y ARTEFACTOS - SABER PRO UAO
"""
import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from src.config import BaseConfig as config    


class TrainModels:

    @staticmethod
    def generar_reportes_y_graficos(target, y_test, preds, modelo):
        nombre_limpio = target.replace(" ", "_").upper()
        figures_dir = os.path.join(config.BASE_DIR, "reports", "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # 1. Gráfico de Importancia de Variables (Guardar en reports/figures/)
        plt.figure(figsize=(8, 5))
        importances = modelo.feature_importances_
        features = [f'COMPONENTE_{i+1}' for i in range(config.PCA_N_COMPONENTS)]
        indices = np.argsort(importances)
        
        plt.title(f"Importancia de Características - {target}", fontsize=12, fontweight='bold')
        plt.barh(range(len(indices)), importances[indices], color='#DEFF9A', edgecolor='#22c55e')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Importancia Relativa')
        plt.tight_layout()
        
        ruta_grafico = os.path.join(figures_dir, f"importancia_{nombre_limpio}.png")
        plt.savefig(ruta_grafico, dpi=300)
        plt.close()
        
        # 2. Gráfico de Predicción vs Valor Real
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, preds, alpha=0.4, color='#22c55e')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
        plt.xlabel('Valores Reales (Icfes/SaberPro)')
        plt.ylabel('Predicciones del Modelo')
        plt.title(f"Dispersión: {target}")
        plt.tight_layout()
        
        ruta_dispersion = os.path.join(figures_dir, f"dispersion_{nombre_limpio}.png")
        plt.savefig(ruta_dispersion, dpi=300)
        plt.close()
        
        return ruta_grafico, ruta_dispersion

    @staticmethod
    def entrenar_modelos():
        print("--- Leyendo dataset procesado tras el ACP... ---")
        ruta_data = config.PROCESSED_DATA_PATH
        if not os.path.exists(ruta_data):
            print(f"Error: No se encuentra el archivo {ruta_data}. Corre primero src/data.py")
            return

        df = pd.read_csv(ruta_data)
        mlflow.set_experiment("Prediccion_Saber_Pro_UAO")
        
        models_dir = os.path.join(config.BASE_DIR, "models")
        figures_dir = os.path.join(config.BASE_DIR, "reports", "figures")
        metrics_dir = os.path.join(config.BASE_DIR, "reports", "metrics")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        
        resumen_metrics = []

        for target in config.TARGET_COLUMNS:
            print(f"\nEntrenando predictor para: {target}...")
            df_clean = df.dropna(subset=[target]).copy()
            if df_clean.empty: continue
                
            feature_cols = [f'COMPONENTE_{i+1}' for i in range(config.PCA_N_COMPONENTS)]
            X = df_clean[feature_cols]
            y = df_clean[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            nombre_limpio = target.replace(" ", "_").upper()
            
            with mlflow.start_run(run_name=f"RF_{nombre_limpio}"):
                modelo = RandomForestRegressor(
                    n_estimators=config.RF_N_ESTIMATORS,
                    max_depth=config.RF_MAX_DEPTH,
                    random_state=config.RANDOM_STATE
                )
                modelo.fit(X_train, y_train)
                preds = modelo.predict(X_test)
                
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                
                mlflow.log_param("n_estimators", config.RF_N_ESTIMATORS)
                mlflow.log_param("max_depth", config.RF_MAX_DEPTH)
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mae", mae)
                
                r_graf, r_disp = TrainModels.generar_reportes_y_graficos(target, y_test, preds, modelo)
                mlflow.log_artifact(r_graf, artifact_path="figures")
                mlflow.log_artifact(r_disp, artifact_path="figures")
                
                mlflow.sklearn.log_model(sk_model=modelo, artifact_path="modelo", registered_model_name=f"RF_{nombre_limpio}")
                
                joblib.dump(modelo, os.path.join(models_dir, f"modelo_{nombre_limpio}.pkl"))
                
                resumen_metrics.append({"Competencia": target, "R2_Score": round(r2, 4), "MAE": round(mae, 2)})

        df_reporte_final = pd.DataFrame(resumen_metrics)
        df_reporte_final.to_csv(os.path.join(metrics_dir, "metricas_consolidadas_saber_pro.csv"), index=False)
        print("\n--- [ÉXITO] Proceso Finalizado. Carpetas 'models' y 'reports' actualizadas ---")

if __name__ == "__main__":
    TrainModels.entrenar_modelos()
