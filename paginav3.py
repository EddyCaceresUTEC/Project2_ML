import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import re
import os

# =======================
# CARGA DE DATOS
# =======================
# Leer X_lda desde el archivo TXT
X_lda = np.loadtxt("X_lda.txt", delimiter=",", skiprows=1)  # Saltar la fila del encabezado
merged_df = pd.read_csv("merged_df_filtrado.csv")

kmeans = KMeans(n_clusters=30, random_state=42)
clusters = kmeans.fit_predict(X_lda)
merged_df["cluster"] = clusters
merged_df["imdbId"] = merged_df["imdbId"].astype(str)
# Convertir valores no numéricos a NaN
merged_df["movieId"] = pd.to_numeric(merged_df["movieId"], errors="coerce")

# Reemplazar NaN con un valor válido (por ejemplo, 0)
merged_df["movieId"] = merged_df["movieId"].fillna(0).astype(float).astype(int).astype(str)
# =======================
# a) BÚSQUEDA POR SIMILITUD VISUAL
# =======================
# Buscar películas similares a una película con un imdbId específico
def buscar_similares(id, n=5, metodo="imdbId"):
    """
    Busca películas similares a una dada basada en su imdbId o movieId.
    :param id: El identificador de la película (imdbId o movieId).
    :param n: Número de películas similares a devolver.
    :param metodo: Método de búsqueda ("imdbId" o "movieId").
    :return: DataFrame con las películas similares.
    """
    if metodo not in ["imdbId", "movieId"]:
        raise ValueError("El método de búsqueda debe ser 'imdbId' o 'movieId'.")

    index_pelicula = merged_df[merged_df[metodo] == str(id)].index[0]
    nn = NearestNeighbors(n_neighbors=n+1).fit(X_lda)
    dists, idxs = nn.kneighbors([X_lda[index_pelicula]])
    columnas_necesarias = ["Title", "Genre", "Year", "Poster"]
    return merged_df.iloc[idxs[0][1:]][columnas_necesarias]

# =======================
# STREAMLIT APP
# =======================
st.set_page_config(page_title="Recomendador de Películas", layout="centered")
st.title("🎬 Recomendador de Películas")
st.write("Selecciona el método de búsqueda y luego ingresa el identificador para obtener recomendaciones.")

# Selección del método de búsqueda
metodo_busqueda = st.radio(
    "Selecciona el método de búsqueda:",
    options=["imdbId", "movieId"],
    index=0  # Por defecto, selecciona imdbId
)

# Selección del número de recomendaciones
numero_recomendaciones = st.selectbox(
    "Selecciona el número de películas similares a mostrar:",
    options=[5, 10, 15],
    index=0  # Por defecto, selecciona 5
)

# Entrada del usuario
identificador_input = st.text_input(f"📌 Ingresa el `{metodo_busqueda}` de la película:")

if identificador_input:
    try:
        # Mostrar información de la película seleccionada
        pelicula_seleccionada = merged_df[merged_df[metodo_busqueda] == str(identificador_input)]
        if pelicula_seleccionada.empty:
            st.error(f"No se encontró ninguna película con el `{metodo_busqueda}`: {identificador_input}")
        else:
            st.write("🎥 Película seleccionada:")
            poster = pelicula_seleccionada.iloc[0]["Poster"]
            title = pelicula_seleccionada.iloc[0]["Title"]
            year = pelicula_seleccionada.iloc[0]["Year"]
            genre_seleccionado = pelicula_seleccionada.iloc[0]["Genre"]
            
            st.image(poster, width=200, caption=f"{title} ({year})")
            st.write(f"Género: {genre_seleccionado}")

            # Obtener recomendaciones
            recomendaciones = buscar_similares(identificador_input, n=numero_recomendaciones, metodo=metodo_busqueda)
            
            st.write("📌 Recomendaciones:")
            
            # Dividir los géneros de la película seleccionada
            generos_seleccionados = set(genre_seleccionado.split("|"))
            
            # Mostrar imágenes de las recomendaciones con colores según coincidencia de géneros
            cols = st.columns(5)
            for idx, (title, genre, year, poster) in enumerate(recomendaciones.values):
                with cols[idx % 5]:
                    st.image(poster, width=120, caption=f"{title} ({year})")
                    
                    # Dividir los géneros de la recomendación
                    generos_recomendacion = set(genre.split("|"))
                    
                    # Determinar el nivel de coincidencia
                    if generos_seleccionados == generos_recomendacion:
                        color = "red"  # Coinciden todos los géneros
                    elif generos_seleccionados & generos_recomendacion:
                        color = "blue"  # Coinciden al menos un género
                    else:
                        color = "black"  # No coinciden géneros
                    
                    # Mostrar el género con el color correspondiente
                    st.markdown(f"<span style='color:{color};'>Género: {genre}</span>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")