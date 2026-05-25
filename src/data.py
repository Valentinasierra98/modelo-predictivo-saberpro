import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import config

def ejecutar_acp_simple():
    print(" Cargando datos y aplicando ACP para la tarea...")
    if not os.path.exists(config.RAW_DATA_PATH):
        print(f" No se encontró el archivo en {config.RAW_DATA_PATH}")
        return

    # 1. Leer el Excel original
    df = pd.read_excel(config.RAW_DATA_PATH)
    
    # 2. Filtrar solo las columnas numéricas para el ACP (excluyendo los 6 targets)
    df_numerico = df.select_dtypes(include=['int64', 'float64']).copy()
    df_numerico = df_numerico.drop(columns=[c for c in config.TARGET_COLUMNS if c in df_numerico.columns])
    
    # 3. Limpieza rápida de nulos con la mediana
    df_numerico = df_numerico.fillna(df_numerico.median())
    
    print(f" Variables numéricas seleccionadas para el ACP: {df_numerico.shape[1]} columnas.")

    # 4. Estandarizar los datos
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_numerico)

    # 5. Aplicar ACP (Reducir a 3 Componentes Principales)
    pca = PCA(n_components=3)
    componentes = pca.fit_transform(datos_escalados)

    # 6. CREAR EL DATASET FINAL (Unir los 3 componentes del ACP con tus 6 variables objetivo)
    df_acp = pd.DataFrame(
        componentes, 
        columns=['COMPONENTE_1', 'COMPONENTE_2', 'COMPONENTE_3']
    )
    
    # Reincorporar los 6 puntajes que debe predecir la tesis
    for target in config.TARGET_COLUMNS:
        df_acp[target] = df[target].values

    # 7. Guardar el resultado final en la carpeta processed
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    df_acp.to_csv(config.PROCESSED_DATA_PATH, index=False)

    # Mostrar resultados analíticos finales en la terminal
    print("\n --- ¡RESULTADOS DEL ACP! ---")
    print(f" Dimensiones de la nueva matriz reducida: {df_acp.shape}")
    print(f" Varianza explicada por cada uno de los 3 componentes: {pca.explained_variance_ratio_}")
    print(f" Varianza total acumulada retenida: {sum(pca.explained_variance_ratio_):.2%}")
    print(f" Archivo de la tarea exportado con éxito en: {config.PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    ejecutar_acp_simple()