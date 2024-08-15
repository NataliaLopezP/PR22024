from etl import etl_pipeline
from ml import (train_regression_models, evaluate_regression_models,
                train_classification_models, evaluate_classification_models,
                train_and_evaluate_kmeans, apply_and_visualize_pca)
from explore import (resumen_descriptivo, contar_paises_por_disparidad,
                     obtener_paises_por_categoria, crear_boxplots,
                     crear_histogramas, comparacion_mpi_urbano_vs_rural,
                     graficar_promedios_mpi_por_region, realizar_anova,
                     crear_pairplot, analizar_correlacion)
import os
import pandas as pd


def main():
    # Definir rutas de archivo
    input_file = os.path.join("Datos", "MPI_national.csv")
    output_file = os.path.join("Datos", "MPI_processed.csv")

    # Ejecutar el pipeline ETL (Carga, Limpieza, Asignación de Continentes, y Cálculo de Poverty Disparity)
    etl_pipeline(input_file, output_file)

    # Cargar los datos procesados
    data = pd.read_csv(output_file)

    # Exploración de datos
    print("\n=== Exploración de Datos ===")
    resumen_descriptivo(data)
    contar_paises_por_disparidad(data)
    obtener_paises_por_categoria(data)
    crear_boxplots(data)
    crear_histogramas(data)
    comparacion_mpi_urbano_vs_rural(data)
    graficar_promedios_mpi_por_region(data)
    realizar_anova(data)
    crear_pairplot(data)
    analizar_correlacion(data)

    # Entrenamiento y evaluación de modelos de regresión
    regression_models, X_test_reg, y_test_reg = train_regression_models(data)
    evaluate_regression_models(regression_models, X_test_reg, y_test_reg)

    # Entrenamiento y evaluación de modelos de clasificación
    classification_models, X_test_class, y_test_class = train_classification_models(
        data)
    evaluate_classification_models(
        classification_models, X_test_class, y_test_class)

    # Entrenamiento y evaluación de K-means
    print("\nAplicando K-Means clustering...")
    clusters = train_and_evaluate_kmeans(data)
    print("Clusters asignados por K-Means:", clusters)

    # Entrenamiento y evaluación de PCA
    print("\nAplicando PCA para reducción de dimensionalidad...")
    pca_result = apply_and_visualize_pca(data)
    print("Datos después de aplicar PCA:", pca_result)


if __name__ == "__main__":
    main()
