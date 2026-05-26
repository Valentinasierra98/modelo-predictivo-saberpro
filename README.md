# Sistema Predictivo de Rendimiento Académico - Saber Pro (UAO)

Este repositorio contiene la implementación de una arquitectura basada en **MLOps (Machine Learning Operations)** para el modelamiento predictivo de factores sociodemográficos y académicos determinantes en las pruebas Saber Pro de la Universidad Autónoma de Occidente (UAO).

El sistema reduce un espacio dimensional complejo mediante Análisis de Componentes Principales (ACP) y entrena un ecosistema multi-estimador optimizado para predecir de forma independiente las múltiples competencias núcleo del examen.

---

##  Arquitectura de Directorios Real (Estándar Industrial)

El proyecto sigue una estructura modular y desacoplada para garantizar el ciclo de vida del modelo, la reproducibilidad y el gobierno de datos:

* **`App/`**: Interfaz gráfica y simulador interactivo web desarrollado en Streamlit para la ejecución de consultas en tiempo real.
* **`data/`**: Gestión integral del ciclo de datos (DataOps).
    * `raw/`: Datos originales y anonimizados de la base de datos de estudiantes (`BASE_TRABAJO_GRADO.xlsx`).
    * `processed/`: Matriz de datos depurada tras la reducción dimensional por Análisis de Componentes Principales (ACP).
    * `monitoring/`: Logs históricos y registros de métricas en producción (`log_inferencia_produccion.csv`) para auditorías en caliente.
* **`models/`**: Repositorio local de almacenamiento de artefactos serializados (los 6 modelos predictivos en formato binario `.pkl`) y banderas de control (`.trained.flag`).
* **`reports/`**: Bitácora científica y metodológica del proyecto.
    * `figures/`: Curvas analíticas de dispersión y gráficos de importancia de características generados en el entrenamiento.
    * `metrics/`: Reportes consolidados y CSV con las métricas de rendimiento ($R^2$ y MAE).
* **`src/`**: Núcleo lógico e ingeniería de software del proyecto.
    * `config.py`: Centralización de variables globales, nombres de competencias y configuraciones del sistema.
    * `data.py`: Pipeline automatizado de extracción, limpieza y procesamiento estadístico (ACP).
    * `Models/train.py`: Pipeline modular de entrenamiento en cascada y tracking de experimentos.
* **`mlflow.db`**: Base de datos relacional local utilizada por el servidor de MLflow para el Gobierno de Modelos (Model Registry) y trazabilidad de versiones.

---

##  Ciclo de Automatización Integrado

Para facilitar el despliegue, la aplicación (`App/app.py`) cuenta con un pipeline inteligente integrado:
1. **Pipeline Auto-Ejecutable:** Si el proyecto se clona por primera vez y no detecta los modelos entrenados en la carpeta `models/`, la aplicación ejecuta automáticamente la ingeniería de datos (ACP) y el entrenamiento de los 6 modelos antes de abrir la interfaz.
2. **Servidor MLflow Automático:** El sistema levanta el servidor local de MLflow en el puerto `5000` en segundo plano y abre la interfaz web en tu navegador de forma automatizada para llevar el gobierno de los experimentos.

---

##  Guía de Replicabilidad Local

Siga estos pasos desde su terminal para clonar, instalar y desplegar el entorno de desarrollo en cualquier computadora:

### 1. Clonar el repositorio
```bash
git clone [https://github.com/Valentinasierra98/modelo-predictivo-saberpro.git](https://github.com/Valentinasierra98/modelo-predictivo-saberpro.git)
cd modelo-predictivo-saberpro