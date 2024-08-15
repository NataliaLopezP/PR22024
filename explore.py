import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Función para obtener resumen descriptivo de las variables numéricas


def resumen_descriptivo(mpi_data):
    descriptive_stats = mpi_data.describe()
    print("Resumen Descriptivo:")
    print(descriptive_stats)
    return descriptive_stats

# Función para contar los países por categoría en 'Poverty Disparity'


def contar_paises_por_disparidad(mpi_data):
    poverty_disparity_count = mpi_data['Poverty Disparity'].value_counts()

    # Crear gráfico de barras con el conteo
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=poverty_disparity_count.index,
                     y=poverty_disparity_count.values)
    plt.title('Conteo de Países por Disparidad de Pobreza')
    plt.xlabel('Poverty Disparity')
    plt.ylabel('Cantidad de Países')

    # Añadir los valores de conteo encima de cada barra
    for index, value in enumerate(poverty_disparity_count.values):
        ax.text(index, value + 0.5, str(value), ha='center')

    plt.show()

# Función para obtener los nombres de los países por cada categoría en 'Poverty Disparity'


def obtener_paises_por_categoria(mpi_data):
    for category in mpi_data['Poverty Disparity'].unique():
        countries_in_category = mpi_data[mpi_data['Poverty Disparity']
                                         == category]['Country'].tolist()
        print(f"\nPaíses en la categoría '{category}':")
        print(countries_in_category)

# Función para crear boxplots para detectar outliers


def crear_boxplots(mpi_data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=mpi_data[['Headcount Ratio Urban', 'Headcount Ratio Rural']])
    plt.title('Boxplot de Ratio Urbano y Rural')
    plt.show()

    sns.boxplot(data=mpi_data[['MPI Urban', 'MPI Rural']])
    plt.title('Boxplot de MPI Urbano y Rural')
    plt.show()

# Función para crear histogramas de MPI Urban y MPI Rural


def crear_histogramas(mpi_data):
    plt.figure(figsize=(12, 6))
    sns.histplot(mpi_data['MPI Urban'], bins=20,
                 kde=True, color='blue', label='MPI Urban')
    sns.histplot(mpi_data['MPI Rural'], bins=20, kde=True,
                 color='orange', label='MPI Rural')
    plt.title('Distribución del MPI en áreas Urbanas y Rurales a nivel Nacional')
    plt.xlabel('MPI')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.show()

# Función para crear scatterplot de comparación entre MPI Urbano y MPI Rural por país


def comparacion_mpi_urbano_vs_rural(mpi_data):
    plt.figure(figsize=(14, 8))
    sns.scatterplot(data=mpi_data, x='MPI Urban', y='MPI Rural',
                    hue='Country', palette='tab20', legend=None)
    plt.title('Comparación entre MPI Urbano y Rural por País')
    plt.xlabel('MPI Urbano')
    plt.ylabel('MPI Rural')
    plt.show()

# Función para graficar los promedios del MPI por región


def graficar_promedios_mpi_por_region(mpi_data):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Continent', y='MPI Urban', data=mpi_data,
                estimator=np.mean, errorbar=None)
    plt.title('Promedio del MPI Urbano por Región')
    plt.ylabel('Promedio MPI Urbano')
    plt.xlabel('Región (Continente)')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Continent', y='MPI Rural', data=mpi_data,
                estimator=np.mean, errorbar=None)
    plt.title('Promedio del MPI Rural por Región')
    plt.ylabel('Promedio MPI Rural')
    plt.xlabel('Región (Continente)')
    plt.xticks(rotation=45)
    plt.show()

# Función para realizar análisis ANOVA entre continentes y MPI Urbano


def realizar_anova(mpi_data):
    mpi_data_filtered = mpi_data.dropna(subset=['Continent'])
    df_anova = mpi_data_filtered[['Continent', 'MPI Urban']]
    grouped_anova = df_anova.groupby('Continent')

    # Obtener los grupos de datos por continente
    group_africa = grouped_anova.get_group('África')['MPI Urban']
    group_asia = grouped_anova.get_group('Asia')['MPI Urban']
    group_europe = grouped_anova.get_group('Europa')['MPI Urban']
    group_north_america = grouped_anova.get_group(
        'América del Norte')['MPI Urban']
    group_south_america = grouped_anova.get_group('América del Sur')[
        'MPI Urban']
    group_oceania = grouped_anova.get_group('Oceanía')['MPI Urban']

    # Realizar el test ANOVA de un solo factor
    F_stat, p_val = stats.f_oneway(
        group_africa,
        group_asia,
        group_europe,
        group_north_america,
        group_south_america,
        group_oceania
    )

    # Resultados del ANOVA
    print("Resultados ANOVA")
    print(f"F = {F_stat}")
    print(f"p = {p_val}")

# Función para crear pairplot del conjunto de datos


def crear_pairplot(mpi_data):
    sns.pairplot(data=mpi_data)
    plt.show()

# Función para realizar análisis de correlación y graficar heatmap


def analizar_correlacion(mpi_data):
    numeric_columns = mpi_data.select_dtypes(include=['float64', 'int64'])

    # Matriz de correlación de variables
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación de las Variables')
    plt.show()
