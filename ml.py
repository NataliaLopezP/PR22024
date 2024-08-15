import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# Función para dividir los datos
def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Función para escalar las características


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Función para entrenar y evaluar modelos de regresión


def train_regression_models(data):
    # Variables independientes y dependientes para regresión
    X = data[['MPI Urban', 'Headcount Ratio Urban', 'Intensity of Deprivation Urban',
              'Headcount Ratio Rural', 'Intensity of Deprivation Rural']]
    y = data['MPI Rural']

    # División de los datos
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Escalar los datos
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Entrenamiento de diferentes modelos de regresión
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    mlp_model = MLPRegressor(hidden_layer_sizes=(
        40,), max_iter=1000, random_state=42)
    mlp_model.fit(X_train_scaled, y_train)

    # Devolver modelos entrenados y datos de prueba
    return {
        "Linear Regression": linear_model,
        "Random Forest": rf_model,
        "MLP Regressor": mlp_model
    }, X_test_scaled, y_test

# Función para evaluar modelos de regresión


def evaluate_regression_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} -> Mean Squared Error: {mse}, R2 Score: {r2}")
        plot_regression_results(y_test, y_pred)

# Función para visualizar los resultados de regresión


def plot_regression_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [
             y_test.min(), y_test.max()], color='red', lw=2)
    plt.title('Actual vs Predicho')
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predicho')
    plt.show()

# Función para entrenar y evaluar modelos de clasificación


def train_classification_models(data):
    # Variables independientes y dependientes para clasificación
    X = data[['MPI Urban', 'Headcount Ratio Urban', 'Intensity of Deprivation Urban',
              'MPI Rural', 'Headcount Ratio Rural']]
    y = data['Poverty Disparity'].map(
        {'Urban Favorable': 0, 'Rural Favorable': 1, 'Equal MPI': 2})

    # División de los datos
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Escalar los datos
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Entrenamiento de diferentes modelos de clasificación
    logistic_model = LogisticRegression(max_iter=400, tol=1e-6)
    logistic_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    mlp_model = MLPClassifier(hidden_layer_sizes=(
        40,), max_iter=1000, random_state=42)
    mlp_model.fit(X_train_scaled, y_train)

    # Devolver modelos entrenados y datos de prueba
    return {
        "Logistic Regression": logistic_model,
        "Random Forest": rf_model,
        "MLP Classifier": mlp_model
    }, X_test_scaled, y_test

# Función para evaluar modelos de clasificación


def evaluate_classification_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} -> Accuracy: {accuracy}")

# Función para aplicar K-means y mostrar los clusters


def apply_kmeans(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters

# Función para aplicar PCA


def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced

# Función para entrenar K-means y visualizar los clusters


def train_and_evaluate_kmeans(data):
    # Usamos las mismas variables que en la clasificación para clustering
    X = data[['MPI Urban', 'Headcount Ratio Urban', 'Intensity of Deprivation Urban',
              'MPI Rural', 'Headcount Ratio Rural']]

    # Escalamos los datos
    X_scaled = StandardScaler().fit_transform(X)

    # Aplicamos K-means
    clusters = apply_kmeans(X_scaled)

    # Visualizamos los clusters
    sns.scatterplot(x=data['MPI Urban'], y=data['MPI Rural'],
                    hue=clusters, palette='viridis')
    plt.title('Clustering de MPI Urbano vs MPI Rural')
    plt.xlabel('MPI Urbano')
    plt.ylabel('MPI Rural')
    plt.show()

    return clusters

# Función para aplicar PCA y visualizar la reducción de dimensionalidad


def apply_and_visualize_pca(data):
    # Usamos las mismas variables para la reducción de dimensionalidad
    X = data[['MPI Urban', 'Headcount Ratio Urban', 'Intensity of Deprivation Urban',
              'MPI Rural', 'Headcount Ratio Rural']]

    # Escalamos los datos
    X_scaled = StandardScaler().fit_transform(X)

    # Aplicamos PCA para reducir a 2 componentes principales
    X_reduced = apply_pca(X_scaled)

    # Visualizamos los componentes principales
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], cmap='viridis')
    plt.title('PCA - Reducción a 2 componentes principales')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()

    return X_reduced
