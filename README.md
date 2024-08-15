# PR22024: Proyecto de Análisis del Índice de Pobreza Multidimensional (MPI)

Este proyecto implementa un flujo de trabajo completo para el análisis de datos del Índice de Pobreza Multidimensional (MPI), que incluye la exploración de datos, el procesamiento de datos (ETL), y la aplicación de modelos de machine learning para regresión, clasificación y clustering.

## Estructura del Proyecto

- `Datos/`: Carpeta que contiene los archivos de datos utilizados en el proyecto.
  - `MPI_national.csv`: Archivo de datos original.
  - `MPI_processed.csv`: Archivo de datos procesado que incluye las nuevas columnas `Continent` y `Poverty Disparity`.

- `etl.py`: Archivo que contiene las funciones de extracción, transformación y carga de datos (ETL).
  - **Funciones**:
    - `load_data`: Carga los datos desde un archivo CSV.
    - `clean_data`: Limpia los datos eliminando valores nulos y duplicados.
    - `assign_continents`: Asigna el continente correspondiente a cada país.
    - `calculate_poverty_disparity`: Calcula la disparidad de pobreza entre áreas urbanas y rurales.

- `explore.py`: Archivo que contiene las funciones para la exploración y visualización de datos.
  - **Funciones**:
    - `resumen_descriptivo`: Muestra el resumen descriptivo de las variables numéricas.
    - `contar_paises_por_disparidad`: Crea un gráfico de barras de la cantidad de países según la disparidad de pobreza.
    - `obtener_paises_por_categoria`: Obtiene los nombres de los países por categoría de disparidad de pobreza.
    - `crear_boxplots`: Crea boxplots para identificar outliers en las variables.
    - `crear_histogramas`: Genera histogramas de las variables `MPI Urban` y `MPI Rural`.
    - `comparacion_mpi_urbano_vs_rural`: Genera un gráfico de dispersión comparando `MPI Urban` vs `MPI Rural`.
    - `graficar_promedios_mpi_por_region`: Genera gráficos de barras para visualizar los promedios de MPI urbano y rural por región.
    - `realizar_anova`: Realiza un análisis ANOVA de un solo factor.
    - `crear_pairplot`: Crea un pairplot de las variables numéricas.
    - `analizar_correlacion`: Genera un heatmap de la matriz de correlación entre las variables.

- `ml.py`: Archivo que contiene las funciones para entrenar y evaluar modelos de machine learning.
  - **Funciones**:
    - Modelos de regresión:
      - `train_linear_regression`
      - `train_random_forest_regressor`
      - `train_mlp_regressor`
    - Modelos de clasificación:
      - `train_logistic_regression`
      - `train_random_forest_classifier`
      - `train_mlp_classifier`
    - Clustering:
      - `apply_kmeans`
      - `apply_pca`
    - Evaluaciones:
      - `evaluate_regression_model`
      - `evaluate_classification_model`
      - `compare_svm_with_pca`

- `main.py`: Script principal que orquesta el flujo de trabajo completo. Ejecuta las fases de ETL, exploración de datos y modelado, llamando a las funciones correspondientes de los archivos `etl.py`, `explore.py` y `ml.py`.

## Dependencias
Las dependencias necesarias incluyen principalmente:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
