# Sistema Predictivo de Rendimiento Académico - Saber Pro (UAO)

Este repositorio contiene la implementación de una arquitectura basada en **MLOps (Machine Learning Operations)** para el modelo predictivo de factores determinantes en las pruebas Saber Pro de los estudiantes de la Universidad xxx.

El sistema reduce un espacio dimensional complejo de variables socioeconómicas e institucionales mediante técnicas avanzadas de Machine Learning, entrenando modelos predictivos optimizados para múltiples competencias.

---

##  Arquitectura de Directorios Real (Estándar Industrial)

El proyecto sigue una estructura modular y desacoplada para garantizar el ciclo de vida del modelo, la reproducibilidad y el gobierno de datos:

* **`App/`**: Interfaz gráfica y simulador interactivo web desarrollado en Streamlit para la ejecución de consultas en tiempo real.
* **`data/`**: Gestión integral del ciclo de datos (DataOps).
    * `raw/`: Datos originales y anonimizados de la base de datos de estudiantes. *[Protegido localmente por peso en .gitignore]*
    * `processed/`: Matriz de datos depurada tras la reducción dimensional por Análisis de Componentes Principales (ACP).
    * `monitoring/`: Logs históricos y registros de métricas en producción (`log_inferencia_produccion.csv`) para auditorías en caliente.
* **`models/`**: Repositorio local de almacenamiento de artefactos serializados (los 6 modelos predictivos en formato binario `.pkl`).
* **`reports/`**: Bitácora científica y metodológica del proyecto.
    * `figures/`: Curvas analíticas de dispersión y gráficos de importancia de características.
    * `metrics/`: Reportes consolidados y CSV con las métricas de rendimiento ($R^2$ y MAE).
* **`src/`**: Núcleo lógico e ingeniería de software del proyecto.
    * `config.py`: Centralización de variables globales, nombres de competencias y configuraciones del sistema.
    * `data.py`: Pipeline automatizado de extracción, limpieza y procesamiento estadístico (ACP).
    * `Models/train.py`: Pipeline modular de entrenamiento en cascada y tracking de experimentos.
* **`mlflow.db`**: Base de datos relacional local utilizada por el servidor de MLflow para el Gobierno de Modelos (Model Registry) y trazabilidad de versiones.

---

##  Guía de Replicabilidad Local

Para replicar este entorno de desarrollo en cualquier computadora, ejecute los siguientes pasos desde la terminal:

### 1. Clonar el repositorio
```bash
git clone [https://github.com/Valentinasierra98/modelo-predictivo-saberpro.git](https://github.com/Valentinasierra98/modelo-predictivo-saberpro.git)
cd modelo-predictivo-saberpro

### 2. Activar el Entorno Virtual (Asegurar dependencias)
En Windows PowerShell, ejecute el siguiente comando para activar el entorno aislado del proyecto:
```powershell
.venv\Scripts\Activate.ps1