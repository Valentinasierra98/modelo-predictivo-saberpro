# Sistema Predictivo de Rendimiento Académico - Saber Pro (UAO)

Este repositorio contiene el laboratorio de desarrollo basado en **MLOps (Machine Learning Operations)** para el modelo predictivo de factores determinantes en las pruebas Saber Pro de los estudiantes de la Universidad XXX.

---

##  Arquitectura de Directorios (Estándar MLOps)

El proyecto sigue una estructura modular para garantizar el ciclo de vida del modelo, la reproducibilidad y el gobierno de datos:

* **`api/`**: Código fuente para el despliegue del servicio web (FastAPI).
* **`app/`**: Interfaz gráfica o prototipo interactivo.
* **`data/`**: Gestión del ciclo de datos (DataOps).
    * `raw/`: Datos originales y anonimizados (`BASE_TRABAJO_GRADO.xlsx`). *[Protegido localmente]*
    * `processed/`: Datasets depurados, imputados y listos para modelamiento.
    * `monitoring/`: Logs y registros de métricas en producción.
* **`models/`**: Binarios de los modelos entrenados (.joblib / .pkl).
* **`reports/`**: Reportes analíticos, curvas de rendimiento y matrices de confusión.
* **`src/`**: Lógica e ingeniería de software del proyecto.
    * `config.py`: Centralización de rutas absolutas y variables globales.
    * `data.py`: Pipeline de extracción, limpieza e imputación de datos.
* **`tests/`**: Pruebas unitarias para asegurar la estabilidad del código.

---

##  Guía de Replicabilidad Local

Para replicar este entorno de desarrollo en cualquier computadora, ejecute los siguientes pasos desde la terminal:

### 1. Clonar el repositorio
```bash
git clone [https://github.com/Valentinasierra98/modelo-predictivo-saberpro.git](https://github.com/Valentinasierra98/modelo-predictivo-saberpro.git)
cd modelo-predictivo-saberpro