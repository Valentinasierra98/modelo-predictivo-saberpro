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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Encontrar la raíz real del proyecto y agregarla al sistema de rutas de Python
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# Ahora importamos config especificando que está dentro de la carpeta src
try:
    from src import config
except ModuleNotFoundError:
    # Si por alguna razón falla el empaquetado, intenta buscarlo de forma directa
    sys.path.append(os.path.join(BASE_DIR, "src"))
    import config

def generar_reportes_y_graficos(target, y_test, preds, modelo):
    nombre_limpio = target.replace(" ", "_").upper()
    
    # 1. Gráfico de Importancia de Variables (Guardar en reports/figures/)
    plt.figure(figsize=(8, 5))
    importances = modelo.feature_importances_
    features = ['COMPONENTE_1', 'COMPONENTE_2', 'COMPONENTE_3']
    indices = np.argsort(importances)
    
    plt.title(f"Importancia de Características - {target}", fontsize=12, fontweight='bold')
    plt.barh(range(len(indices)), importances[indices], color='#DEFF9A', edgecolor='#22c55e')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Importancia Relativa')
    plt.tight_layout()
    
    ruta_grafico = f"reports/figures/importancia_{nombre_limpio}.png"
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
    
    ruta_dispersion = f"reports/figures/dispersion_{nombre_limpio}.png"
    plt.savefig(ruta_dispersion, dpi=300)
    plt.close()
    
    return ruta_grafico, ruta_dispersion

def entrenar_modelos():
    print("--- Leyendo dataset procesado tras el ACP... ---")
    # Ruta adaptada a tu estructura local de datos
    ruta_data = "data/processed/data_limpia.csv"
    if not os.path.exists(ruta_data):
        print(f"Error: No se encuentra el archivo {ruta_data}. Corre primero src/data.py")
        return

    df = pd.read_csv(ruta_data)
    mlflow.set_experiment("Prediccion_Saber_Pro_UAO")
    
    # Crear carpetas correctas en la raíz
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)
    
    resumen_metrics = []

    for target in config.TARGET_COLUMNS:
        print(f"\nEntrenando predictor para: {target}...")
        df_clean = df.dropna(subset=[target]).copy()
        if df_clean.empty: continue
            
        X = df_clean[['COMPONENTE_1', 'COMPONENTE_2', 'COMPONENTE_3']]
        y = df_clean[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        nombre_limpio = target.replace(" ", "_").upper()
        
        with mlflow.start_run(run_name=f"RF_{nombre_limpio}"):
            modelo = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            modelo.fit(X_train, y_train)
            preds = modelo.predict(X_test)
            
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 8)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            
            r_graf, r_disp = generar_reportes_y_graficos(target, y_test, preds, modelo)
            mlflow.log_artifact(r_graf, artifact_path="figures")
            mlflow.log_artifact(r_disp, artifact_path="figures")
            
            mlflow.sklearn.log_model(sk_model=modelo, artifact_path="modelo", registered_model_name=f"RF_{nombre_limpio}")
            
            # Guardar el binario en la carpeta models/ de la raíz para la App
            joblib.dump(modelo, f"models/modelo_{nombre_limpio}.pkl")
            
            resumen_metrics.append({"Competencia": target, "R2_Score": round(r2, 4), "MAE": round(mae, 2)})

    df_reporte_final = pd.DataFrame(resumen_metrics)
    df_reporte_final.to_csv("reports/metrics/metricas_consolidadas_saber_pro.csv", index=False)
    print("\n--- [ÉXITO] Proceso Finalizado. Carpetas 'models' y 'reports' actualizadas ---")

if __name__ == "__main__":
    entrenar_modelos()