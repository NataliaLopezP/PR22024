import pandas as pd

# Diccionario de continentes
continent_dict = {
    "África": [
        "Tunisia", "Algeria", "Morocco", "Egypt", "South Africa", "Swaziland",
        "Zimbabwe", "Lesotho", "Gabon", "Sao Tome and Principe", "Ghana", "Kenya",
        "Togo", "Comoros", "Namibia", "Cameroon", "Malawi", "Rwanda", "Madagascar",
        "Mauritania", "Zambia", "Nigeria", "Uganda", "Sudan", "Senegal",
        "Cote d'Ivoire", "Guinea", "Gambia", "Mali", "Mozambique", "Benin",
        "Guinea-Bissau", "Burundi", "Burkina Faso", "Congo, Democratic Republic of the",
        "Ethiopia", "Sierra Leone", "Niger", "Central African Republic", "Liberia",
        "Somalia", "Chad", "South Sudan", "Djibouti"
    ],
    "Asia": [
        "Kazakhstan", "Kyrgyzstan", "Turkmenistan", "Uzbekistan", "Tajikistan",
        "Armenia", "Mongolia", "Thailand", "China", "Viet Nam", "Cambodia",
        "Myanmar", "Philippines", "Indonesia", "Lao People's Democratic Republic",
        "India", "Pakistan", "Bangladesh", "Nepal", "Maldives", "Sri Lanka",
        "Bhutan", "Afghanistan", "Yemen", "Jordan", "Iraq", "Syrian Arab Republic"
    ],
    "Europa": [
        "Serbia", "Macedonia, The former Yugoslav Republic of", "Moldova, Republic of",
        "Montenegro", "Bosnia and Herzegovina", "Albania", "Ukraine"
    ],
    "América del Norte": [
        "Mexico", "Belize"
    ],
    "América del Sur": [
        "Guyana", "Ecuador", "Colombia", "El Salvador", "Peru", "Nicaragua",
        "Brazil", "Bolivia, Plurinational State of", "Honduras", "Dominican Republic",
        "Guatemala", "Trinidad and Tobago", "Barbados", "Jamaica", "Haiti",
        "Saint Lucia"
    ],
    "Oceanía": [
        "Vanuatu"
    ]
}


def load_data(file_path):
    # Cargar los datos desde un archivo CSV
    return pd.read_csv(file_path)


def clean_data(df):
    # Limpiar los datos eliminando filas con valores faltantes
    df_cleaned = df.dropna()
    return df_cleaned


def assign_continents(df, continents=continent_dict):
    # Función interna para asignar continentes
    def get_continent(country):
        for continent, countries in continents.items():
            if country in countries:
                return continent
        return None

    # Aplicar la función a la columna 'Country'
    df['Continent'] = df['Country'].apply(get_continent)
    return df


def calculate_poverty_disparity(df):
    # Función para calcular la disparidad de pobreza entre áreas urbanas y rurales
    def classify_disparity(row):
        if row['MPI Urban'] < row['MPI Rural']:
            return 'Urban Favorable'
        elif row['MPI Urban'] > row['MPI Rural']:
            return 'Rural Favorable'
        else:
            return 'Equal MPI'

    # Crear una nueva columna 'Poverty Disparity' basándose en la comparación entre MPI Urban y MPI Rural
    df['Poverty Disparity'] = df.apply(classify_disparity, axis=1)
    return df


def save_processed_data(df, output_path):
    # Guardar los datos procesados en un archivo CSV
    df.to_csv(output_path, index=False)


def etl_pipeline(file_path, output_path):
    # Cargar los datos
    data = load_data(file_path)

    # Limpiar los datos
    data_cleaned = clean_data(data)

    # Asignar continentes
    data_with_continents = assign_continents(data_cleaned, continent_dict)

    # Calcular disparidad de pobreza
    data_with_disparity = calculate_poverty_disparity(data_with_continents)

    # Guardar los datos procesados con ambas columnas en el mismo archivo CSV
    save_processed_data(data_with_disparity, output_path)
