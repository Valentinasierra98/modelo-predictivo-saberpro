import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ==========================================
#  CONFIGURACIÓN DE RUTAS ROBUSTAS (RAÍZ DEL PROYECTO)
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CARPETA_MODELOS = os.path.join(BASE_DIR, "models")  # Apunta a la carpeta unificada 'models'
RUTA_MONITOREO = os.path.join(BASE_DIR, "data", "monitoring", "log_inferencia_produccion.csv")

# ==========================================
#  FUNCIÓN DE PERSISTENCIA (DATAOPS / MONITORING)
# ==========================================
def registrar_prediccion_monitoreo(c1, c2, c3, pred_global, pred_rc, pred_lc, pred_cc, pred_ing, pred_ce):
    """
    Guarda en caliente los datos calculados del ACP y las predicciones 
    generadas dentro de la carpeta data/monitoring.
    """
    registro = {
        "FECHA_LOG": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ACP_COMPONENTE_1": round(c1, 4),
        "ACP_COMPONENTE_2": round(c2, 4),
        "ACP_COMPONENTE_3": round(c3, 4),
        "PRED_PUNT_GLOBAL": round(pred_global, 1),
        "PRED_RAZ_CUANT": round(pred_rc, 1),
        "PRED_LECT_CRIT": round(pred_lc, 1),
        "PRED_COMP_CIUD": round(pred_cc, 1),
        "PRED_INGLES": round(pred_ing, 1),
        "PRED_COM_ESC": round(pred_ce, 1)
    }
    
    df_registro = pd.DataFrame([registro])
    
    # Asegurar que la carpeta exista antes de intentar escribir
    os.makedirs(os.path.dirname(RUTA_MONITOREO), exist_ok=True)
    
    # Si el archivo no existe, lo crea con encabezados. Si existe, concatena abajo (append)
    if not os.path.exists(RUTA_MONITOREO):
        df_registro.to_csv(RUTA_MONITOREO, index=False)
    else:
        df_registro.to_csv(RUTA_MONITOREO, mode='a', header=False, index=False)


# Configuración de la plataforma
st.set_page_config(page_title="Predictor Saber Pro", layout="wide", page_icon="🎓")

# CSS Profesional Unificado (Tu diseño original)
st.markdown("""
    <style>
    .model-box {
        background-color: #1A261A;
        border: 1px solid #22c55e;
        border-radius: 10px;
        padding: 15px 20px;
        color: #22c55e;
        font-weight: bold;
        font-size: 1.05rem;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 15px;
        padding: 22px;
        border: 1px solid #DEFF9A;
        box-shadow: 0 4px 15px rgba(222, 255, 154, 0.08);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-title { color: #DEFF9A; font-size: 1.1rem; font-weight: bold; margin-bottom: 8px; text-transform: uppercase; }
    .metric-value { color: white; font-size: 2.6rem; font-weight: 800; margin-bottom: 3px; }
    .metric-meta { color: #888888; font-size: 0.85rem; font-weight: 600; }
    .logo-text { color: #DEFF9A; font-size: 24px; font-weight: 800; font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Predictor de Rendimiento Saber Pro - UAO")
st.markdown("### Modelamiento Predictivo Basado en Variables Sociodemográficas e Historial Académico")

st.markdown("""
    <div class="model-box">
         <strong>Algoritmo Core Electo:</strong> Multi-Ecosistema Random Forest | Inferencia Local en Tiempo Real
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
#  CARGA MULTI-MODELO DESDE CARPETA REAL
# ==========================================
@st.cache_resource  # Carga los modelos una sola vez en memoria para optimizar velocidad
def cargar_ecosistema_modelos():
    materias_claves = [
        "PUNT_GLOBAL", 
        "MOD_RAZONA_CUANTITAT_PUNT", 
        "MOD_LECTURA_CRITICA_PUNT", 
        "MOD_COMPETEN_CIUDADA_PUNT", 
        "MOD_INGLES_PUNT", 
        "MOD_COMUNI_ESCRITA_PUNT"
    ]
    
    dict_modelos = {}
    todos_existen = True
    
    # Intentar cargar cada uno de los 6 modelos individuales usando la ruta absoluta robusta
    for materia in materias_claves:
        nombre_archivo = f"modelo_{materia}.pkl"
        ruta_completa = os.path.join(CARPETA_MODELOS, nombre_archivo)
        
        if os.path.exists(ruta_completa):
            dict_modelos[materia] = joblib.load(ruta_completa)
        else:
            todos_existen = False
            st.error(f" No se encontró '{nombre_archivo}' dentro de la carpeta '{CARPETA_MODELOS}'.")
            
    if todos_existen and len(dict_modelos) == 6:
        return dict_modelos
    return None

# Intentamos cargar el diccionario con los 6 Random Forest
modelos_cargados = cargar_ecosistema_modelos()

# Sidebar: Variables solicitadas (Simulación de entrada del usuario)
with st.sidebar:
    st.markdown('<div class="logo-text"> University Analytics</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.header(" Perfil del Estudiante")
    st.write("Configure los factores sociodemográficos y académicos reales:")
    
    estrato = st.slider(" Estrato Socioeconómico", 1, 6, 3, step=1)
    ingresos = st.slider(" Ingresos Familiares (En SMMLV)", 1, 10, 2, step=1)
    
    st.markdown("---")
    icfes = st.slider(" Puntaje ICFES Saber 11 (Global)", 0, 500, 280, step=5)
    promedio_u = st.slider(" Promedio Acumulado Universitario", 0.0, 5.0, 3.8, step=0.1)
    
    st.markdown("---")
    horas_estudio = st.slider(" Horas de estudio autónomo (Semanales)", 0, 40, 12, step=2)

# Métricas estáticas de validación del modelo (Establecidas en tu entrenamiento)
metricas_fijas = {
    "PUNT_GLOBAL": {"titulo": "PUNTAJE GLOBAL", "r2": "73%", "mae": "7.8 Pts"},
    "MOD_RAZONA_CUANTITAT_PUNT": {"titulo": "RAZONAMIENTO CUANTITATIVO", "r2": "48%", "mae": "18.0 Pts"},
    "MOD_LECTURA_CRITICA_PUNT": {"titulo": "LECTURA CRÍTICA", "r2": "57%", "mae": "15.4 Pts"},
    "MOD_COMPETEN_CIUDADA_PUNT": {"titulo": "COMPETENCIAS CIUDADANAS", "r2": "52%", "mae": "17.8 Pts"},
    "MOD_INGLES_PUNT": {"titulo": "INGLÉS", "r2": "39%", "mae": "17.8 Pts"},
    "MOD_COMUNI_ESCRITA_PUNT": {"titulo": "COMUNICACIÓN ESCRITA", "r2": "27%", "mae": "23.2 Pts"}
}

# Despliegue matricial de resultados (2 filas x 3 columnas)
rows = [st.columns(3), st.columns(3)]
claves_materias = list(metricas_fijas.keys())

if modelos_cargados is not None:
    # Coordenadas calculadas sintéticamente basadas en los pesos del ACP
    c1 = (icfes * 0.015) + (promedio_u * 0.4) + (horas_estudio * 0.02) - 2.1
    c2 = (estrato * -0.3) + (ingresos * -0.2) + (promedio_u * 0.1) + 0.5
    c3 = (horas_estudio * 0.1) - (icfes * 0.002)
    
    datos_entrada_acp = [[c1, c2, c3]]
    
    # Inicializar variables temporales para el registro de monitoreo
    pred_global = pred_rc = pred_lc = pred_cc = pred_ing = pred_ce = 0.0
    
    # Iterar sobre las 6 materias para predecir de forma independiente
    for i, materia_key in enumerate(claves_materias):
        resultado_prediccion = modelos_cargados[materia_key].predict(datos_entrada_acp)[0]
        
        # Validación de límites para que los puntajes no se salgan del estándar (0 a 300 o 0 a 500)
        if materia_key == "PUNT_GLOBAL":
            resultado_prediccion = min(max(resultado_prediccion, 0), 500)
            pred_global = resultado_prediccion
        else:
            resultado_prediccion = min(max(resultado_prediccion, 0), 300)
            if i == 1: pred_rc = resultado_prediccion
            elif i == 2: pred_lc = resultado_prediccion
            elif i == 3: pred_cc = resultado_prediccion
            elif i == 4: pred_ing = resultado_prediccion
            elif i == 5: pred_ce = resultado_prediccion
            
        col_idx = i % 3
        row_idx = 0 if i < 3 else 1
        
        info_materia = metricas_fijas[materia_key]
        
        with rows[row_idx][col_idx]:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{info_materia['titulo']}</div>
                    <div class="metric-value">{resultado_prediccion:.1f}</div>
                    <div class="metric-meta">Confianza (R²): {info_materia['r2']} &nbsp;|&nbsp; Error: ±{info_materia['mae']}</div>
                </div>
            """, unsafe_allow_html=True)
            
    # Botón profesional para accionar el log en caliente
    if st.button(" Registrar consulta en bitácora de producción"):
        registrar_prediccion_monitoreo(c1, c2, c3, pred_global, pred_rc, pred_lc, pred_cc, pred_ing, pred_ce)
        st.success(f" Simulación registrada de manera exitosa en: {os.path.relpath(RUTA_MONITOREO, BASE_DIR)}")

else:
    st.warning(f" Esperando a que el backend verifique los 6 archivos '.pkl' en la carpeta '{CARPETA_MODELOS}' para activar la inferencia.")

st.markdown("---")
st.success(" MLOps e Interfaz Sincronizados: El backend ejecuta inferencia directa mapeando las variables socioeconómicas a los 3 Componentes Principales en tiempo real.")